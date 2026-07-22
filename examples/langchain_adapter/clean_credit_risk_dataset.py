"""将信贷审批意见清洗为 GEPA JSONL 数据.

清洗器只提取审批官意见中的“项目风险点”章节. 智能体输入保持最小化, 通常
只有企业名称; 专家章节存入 ``data``, 仅供评估器和反思步骤使用.

示例:
    uv run --no-sync python examples/langchain_adapter/clean_credit_risk_dataset.py \
      --config examples/langchain_adapter/deepagents_gepa_configs/credit_approval.toml \
      --input-dir /path/to/opinions \
      --output examples/langchain_adapter/deepagents_gepa_credit_approval_project/evals/project_risk_sections.jsonl

可选传入 ``--extraction-model``, 由 LLM 在真实工具清单约束下提取 checkpoint、
checkpoint 到证据类别的映射, 以及每类证据允许使用的 ``tool_names``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
SECTION_TITLES = (
    "项目风险点",
    "主要风险点",
    "项目主要风险",
    "风险分析",
    "风险因素",
    "风险提示",
    "风险及防范措施",
)
SECTION_START_RE = re.compile(
    r"^\s*(?:[一二三四五六七八九十]+[\u3001.\uff0e]\s*)?"
    r"(?P<title>项目风险点|主要风险点|项目主要风险|风险分析|风险因素|风险提示|风险及防范措施)\s*$"
)
TOP_LEVEL_HEADING_RE = re.compile(r"^\s*[一二三四五六七八九十]+[\u3001.\uff0e]\s*\S+")
RISK_HEADING_RE = re.compile(
    r"^\s*(?:\d+|[一二三四五六七八九十]+)[\u3001.\uff0e]\s*(?P<label>[^\n]{2,80}?风险[^\n]*)\s*$",
    re.M,
)
ACTION_ONLY_CHECKPOINT_PATTERN = re.compile(
    r"(?:审批建议|授信建议|授信压降|额度调整|放款条件|提款条件|贷后方案|"
    r"回款监管必要性|追加担保|现金保证金|否决建议)"
)
RISK_SEMANTIC_PATTERN = re.compile(r"(?:风险|压力|恶化|异常|缺口|不确定|失效|不足|波动|背离)")
FILENAME_NOISE_PATTERN = re.compile(
    r"(?:风险评价意见书|风险评价意见|风险评价|授信审批意见书|授信审批|授信调查报告|授信调查|"
    r"尽调报告|项目风险点|审批意见)(?:[_\-\s]*(?:最终版|修订版|定稿|v?\d+(?:\.\d+)?))?$",
    re.I,
)
GENERIC_FILENAME_STEMS = frozenset({"风险评价意见书", "风险评价", "授信审批", "尽调报告", "项目风险点", "审批意见"})

MetadataExtractor = Callable[
    [Path, str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, str]]],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class CleanedRiskSection:
    source_file: Path
    company_name: str
    section_title: str
    section_text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="包含审批意见文件的目录.")
    source.add_argument("--input-file", action="append", help="单个审批意见文件, 可重复传入.")
    parser.add_argument("--config", required=True, help="用于运行 agent 的 deepagents_gepa TOML 配置.")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件.")
    parser.add_argument("--extraction-model", help="可选的 LangChain 模型名, 用于结构化提取 metadata.")
    parser.add_argument("--extraction-model-kwargs", type=json.loads, default={})
    parser.add_argument("--id-prefix", default="credit_case")
    parser.add_argument("--no-metadata", action="store_true", help="不生成启发式元数据.")
    parser.add_argument("--overwrite", action="store_true", help="输出文件已存在时覆盖.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} 已存在, 请传入 --overwrite 后再覆盖.")
    source_files = collect_source_files(args)
    tool_inventory = load_agent_tool_inventory(args.config)
    metadata_extractor = (
        make_llm_metadata_extractor(args.extraction_model, args.extraction_model_kwargs)
        if args.extraction_model
        else None
    )
    records = []
    for index, source_file in enumerate(source_files, start=1):
        cleaned = clean_one_file(
            source_file,
            include_metadata=not args.no_metadata,
            tool_inventory=tool_inventory,
            metadata_extractor=metadata_extractor,
        )
        if cleaned is None:
            continue
        record: dict[str, Any] = {
            "id": f"{args.id_prefix}_{index:04d}",
            "input": cleaned.company_name,
            "data": cleaned.section_text,
        }
        if not args.no_metadata:
            record["metadata"] = cleaned.metadata
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    print(f"已将 {len(records)} 条数据写入 {output}")


def collect_source_files(args: argparse.Namespace) -> list[Path]:
    if args.input_file:
        return [Path(item).expanduser().resolve() for item in args.input_file]
    input_dir = Path(args.input_dir).expanduser().resolve()
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and not path.name.startswith("~$") and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def clean_one_file(
    source_file: Path,
    *,
    include_metadata: bool = True,
    tool_inventory: Sequence[Mapping[str, str]] = (),
    metadata_extractor: MetadataExtractor | None = None,
) -> CleanedRiskSection | None:
    text = read_source_text(source_file)
    extracted = extract_project_risk_section(text)
    if extracted is None:
        return None
    section_title, section_text = extracted
    risk_points = extract_risk_points(section_text)
    extracted_metadata = (
        dict(metadata_extractor(source_file, section_text, risk_points, tool_inventory))
        if metadata_extractor is not None
        else {}
    )
    company_name = infer_company_name(source_file, text, fallback_name=extracted_metadata.get("company_name"))
    metadata: dict[str, Any] = {
        "source_file": source_file.name,
        "section": section_title,
    }
    if include_metadata:
        checkpoints = normalize_checkpoints(extracted_metadata.get("checkpoints")) or build_checkpoints(risk_points)
        expectations = normalize_trace_expectations(
            extracted_metadata.get("trace_expectations"),
            tool_inventory,
        ) or build_trace_expectations(risk_points, tool_inventory=tool_inventory)
        metadata["trace_expectations"] = expectations
        linked_checkpoints = link_checkpoints_to_evidence(checkpoints, risk_points, expectations)
        metadata["checkpoints"] = linked_checkpoints
        metadata.update(tool_coverage_metadata(linked_checkpoints, expectations))
    return CleanedRiskSection(
        source_file=source_file,
        company_name=company_name,
        section_title=section_title,
        section_text=section_text,
        metadata=metadata,
    )


def enrich_existing_record(
    record: Mapping[str, Any],
    tool_inventory: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Migrate an existing JSONL row to dataset-level rubric and evidence mappings."""
    enriched = dict(record)
    enriched.pop("rubric", None)
    metadata = dict(enriched.get("metadata") or {})
    risk_points = extract_risk_points(str(enriched.get("data") or ""))
    checkpoints = normalize_checkpoints(metadata.get("checkpoints")) or build_checkpoints(risk_points)
    expectations = normalize_trace_expectations(metadata.get("trace_expectations"), tool_inventory)
    if not expectations:
        expectations = build_trace_expectations(risk_points, tool_inventory=tool_inventory)
    metadata["trace_expectations"] = expectations
    linked_checkpoints = link_checkpoints_to_evidence(checkpoints, risk_points, expectations)
    metadata["checkpoints"] = linked_checkpoints
    metadata.update(tool_coverage_metadata(linked_checkpoints, expectations))
    metadata.pop("risk_points", None)
    enriched["metadata"] = metadata
    return enriched


def read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return read_docx_text(path)
    if suffix == ".pdf":
        return read_pdf_text(path)
    raise ValueError(f"不支持的文件类型: {path}")


def read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs)


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("读取 PDF 需要 pypdf; 请安装 pypdf, 或先将文件转换为 .txt.") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_project_risk_section(text: str) -> tuple[str, str] | None:
    lines = normalize_text_lines(text)
    start_index: int | None = None
    section_title = ""
    for index, line in enumerate(lines):
        match = SECTION_START_RE.match(line)
        if match:
            start_index = index
            section_title = match.group("title")
            break
    if start_index is None:
        return None

    section_lines = [lines[start_index]]
    for line in lines[start_index + 1 :]:
        if TOP_LEVEL_HEADING_RE.match(line) and not any(title in line for title in SECTION_TITLES):
            break
        section_lines.append(line)
    section_text = "\n".join(section_lines).strip()
    return section_title, section_text


def normalize_text_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return [line.strip() for line in normalized.split("\n") if line.strip()]


def infer_company_name(path: Path, text: str, fallback_name: Any = None) -> str:
    filename_name = company_name_from_filename(path)
    if filename_name:
        return filename_name
    if str(fallback_name or "").strip():
        return str(fallback_name).strip()
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9\uff08\uff09()]{2,80}(?:公司|集团|企业))", text)
    if match:
        return match.group(1)
    return path.stem


def company_name_from_filename(path: Path) -> str | None:
    stem = unicodedata.normalize("NFKC", path.stem).strip()
    candidate = FILENAME_NOISE_PATTERN.sub("", stem).strip(" _-—")
    if not candidate or candidate in GENERIC_FILENAME_STEMS:
        return None
    company_match = re.search(
        r"([\u4e00-\u9fa5A-Za-z0-9()]{2,80}(?:有限责任公司|股份有限公司|有限公司|集团))", candidate
    )
    return company_match.group(1) if company_match else candidate


def extract_risk_points(section_text: str) -> list[dict[str, Any]]:
    matches = [
        match
        for match in RISK_HEADING_RE.finditer(section_text)
        if normalize_label(match.group("label")) not in SECTION_TITLES
    ]
    if not matches:
        return [{"label": "项目风险点", "text": section_text, "keywords": extract_keywords(section_text)}]
    points: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section_text)
        label = normalize_label(match.group("label"))
        body = section_text[start:end].strip()
        points.append({"label": label, "text": body, "keywords": extract_keywords(f"{label}\n{body}")})
    return points


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", "", label).strip()


KEYWORD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("行业周期信息获取", ("行业", "周期", "价格", "库存", "需求", "钢铁", "地产", "基建")),
    ("盈利质量信息获取", ("盈利", "利润", "毛利", "净利", "收入", "现金流", "回款")),
    ("债务结构信息获取", ("负债", "债务", "短期借款", "长期借款", "票据", "短贷长投", "财务费用", "融资")),
    ("集团穿透信息获取", ("集团", "合并报表", "子公司", "贸易板块", "生产基地", "担保", "互保", "隐性融资")),
    ("合规处罚信息获取", ("环保", "安全生产", "行政处罚", "监管", "执法", "安监", "处罚")),
    ("司法工商信息获取", ("诉讼", "执行", "司法", "股权", "实控人", "工商", "变更")),
    ("抵质押担保信息获取", ("抵押", "质押", "担保", "顺位", "评估", "权属")),
    ("客户交易信息获取", ("客户", "集中", "合同", "订单", "发票", "流水", "应收")),
    ("财政项目回款信息获取", ("财政", "拨付", "审计确权", "付款计划", "政府工程")),
    ("保险补贴信息获取", ("保险", "除外责任", "政府补贴", "补贴")),
)


def load_agent_tool_inventory(config_path: str | Path) -> list[dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.langchain_adapter.deep_agent_skill_directory import (
        build_candidate_from_deep_agent_project,
        load_deepagents_gepa_config,
    )

    config = load_deepagents_gepa_config(config_path)
    project = build_candidate_from_deep_agent_project(config)
    inventory: list[dict[str, str]] = []
    for key, description in project.candidate.items():
        if ":tool:" not in key or not key.endswith(":description"):
            continue
        owner, tool_part = key.split(":tool:", 1)
        inventory.append(
            {
                "owner": owner,
                "name": tool_part.removesuffix(":description"),
                "description": description,
            }
        )
    return inventory


def make_llm_metadata_extractor(model_name: str, model_kwargs: Mapping[str, Any]) -> MetadataExtractor:
    from langchain.chat_models import init_chat_model

    model = init_chat_model(model_name, **dict(model_kwargs))

    def extract(
        source_file: Path,
        section_text: str,
        risk_points: Sequence[Mapping[str, Any]],
        tool_inventory: Sequence[Mapping[str, str]],
    ) -> Mapping[str, Any]:
        prompt = (
            "你是中国规上企业信贷风险评价数据清洗器。只整理专家意见, 不补充材料中没有的企业事实。\n"
            "从项目风险点章节提取: company_name、checkpoints、trace_expectations。\n"
            "每个 checkpoint 只表达一个可独立评分的风险机制。标题中含“与、和、及”等复合关系时, 如果正文包含"
            "可分别验证的机制, 应拆成多个 checkpoint; 不要把盈利波动和盈利质量、债务规模和债务结构等合成"
            "一个全有或全无的 checkpoint。每个 checkpoint 保留风险点名称和少量中文别名, 并通过 "
            "evidence_expectations 关联到一项或多项 "
            "trace expectation; 默认 evidence_mode=all。trace expectation 描述取得该证据的意图, 并且 "
            "tool_names 只能从给定工具清单中选择。工具描述明确为政策查询、写入/保存或不查询企业事实时, "
            "不得把它当成企业证据获取工具。没有对应工具就省略 tool_names, 不能猜工具能力。\n"
            "不得把单个企业材料中的计算结果、比例或金额改写成通用阈值。只有材料明确表述为政策、制度或"
            "审批规则的阈值才能保留为规则, 其他情形使用趋势、同业、历史或结构比较。\n"
            "排除授信额度、放款条件、审批结论、贷后措施等纯行动项。\n"
            "仅返回 JSON:"
            '{"company_name":"", "checkpoints":[{"label":"", "keywords":[], '
            '"evidence_expectations":[], "evidence_mode":"all"}], '
            '"trace_expectations":[{"label":"", "tool_intent_keywords":[], "tool_names":[]}]}\n\n'
            f"文件名: {source_file.name}\n"
            f"启发式风险点: {json.dumps(list(risk_points), ensure_ascii=False)}\n"
            f"真实工具清单: {json.dumps(list(tool_inventory), ensure_ascii=False)}\n\n"
            f"项目风险点章节:\n{section_text}"
        )
        response = model.invoke(prompt)
        return parse_json_object(message_content(response))

    return extract


def message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, Mapping) else str(item) for item in content)
    return str(content)


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("LLM metadata extraction must return a JSON object")
    return payload


def extract_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    compact = re.sub(r"\s+", "", text)
    for _label, group_keywords in KEYWORD_GROUPS:
        for keyword in group_keywords:
            if keyword in compact and keyword not in keywords:
                keywords.append(keyword)
    label_terms = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}", compact[:120])
    for term in label_terms:
        if term not in keywords:
            keywords.append(term)
        if len(keywords) >= 8:
            break
    return keywords[:8]


def build_checkpoints(risk_points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = []
    for point in risk_points:
        label = str(point.get("label") or "项目风险点")
        if is_action_only_checkpoint(label):
            continue
        keywords = [str(item) for item in point.get("keywords", []) if str(item).strip()]
        checkpoints.append({"label": label, "keywords": keywords or [label]})
    return checkpoints


def build_trace_expectations(
    risk_points: Sequence[Mapping[str, Any]],
    *,
    tool_inventory: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, Any]]:
    labels: dict[str, list[str]] = {}
    joined_points = "\n".join(
        f"{point.get('label', '')}\n{point.get('text', '')}"
        for point in risk_points
        if not is_action_only_checkpoint(str(point.get("label") or ""))
    )
    compact = re.sub(r"\s+", "", joined_points)
    for label, keywords in KEYWORD_GROUPS:
        matched = [keyword for keyword in keywords if keyword in compact]
        if matched:
            labels[label] = matched
    expectations = [
        {"label": label, "tool_intent_keywords": keywords}
        for label, keywords in sorted(labels.items(), key=lambda item: item[0])
    ]
    return attach_tool_names(expectations, tool_inventory)


def normalize_checkpoints(raw_checkpoints: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_checkpoints, list):
        return []
    checkpoints = []
    for raw in raw_checkpoints:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label") or "").strip()
        if not label or is_action_only_checkpoint(label):
            continue
        keywords = _string_list(raw.get("keywords")) or [label]
        checkpoint = {"label": label, "keywords": keywords[:8]}
        evidence_expectations = _string_list(raw.get("evidence_expectations"))
        if evidence_expectations:
            checkpoint["evidence_expectations"] = evidence_expectations
            checkpoint["evidence_mode"] = "any" if str(raw.get("evidence_mode")).lower() == "any" else "all"
        checkpoints.append(checkpoint)
    return checkpoints


def normalize_trace_expectations(
    raw_expectations: Any,
    tool_inventory: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    if not isinstance(raw_expectations, list):
        return []
    allowed_tools = {str(item.get("name") or "") for item in tool_inventory if tool_can_supply_entity_evidence(item)}
    expectations = []
    for raw in raw_expectations:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label") or "").strip()
        if not label:
            continue
        keywords = _string_list(raw.get("tool_intent_keywords") or raw.get("keywords")) or [label]
        expectation: dict[str, Any] = {"label": label, "tool_intent_keywords": keywords[:10]}
        tool_names = [name for name in _string_list(raw.get("tool_names")) if name in allowed_tools]
        if tool_names:
            expectation["tool_names"] = tool_names
        expectations.append(expectation)
    return attach_tool_names(expectations, tool_inventory)


def attach_tool_names(
    expectations: Sequence[Mapping[str, Any]],
    tool_inventory: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    attached = []
    for raw in expectations:
        expectation = dict(raw)
        if not expectation.get("tool_names"):
            keywords = _string_list(expectation.get("tool_intent_keywords"))
            scored_tools = []
            for tool_item in tool_inventory:
                if not tool_can_supply_entity_evidence(tool_item):
                    continue
                description = str(tool_item.get("description") or "")
                matches = sum(1 for keyword in keywords if keyword_matches(description, keyword))
                required = min(2, len(keywords))
                if required and matches >= required:
                    scored_tools.append((matches, str(tool_item.get("name") or "")))
            if scored_tools:
                best_score = max(score for score, _name in scored_tools)
                expectation["tool_names"] = sorted(
                    {name for score, name in scored_tools if score == best_score and name}
                )
        attached.append(expectation)
    return attached


def tool_can_supply_entity_evidence(tool_item: Mapping[str, str]) -> bool:
    description = str(tool_item.get("description") or "")
    return not bool(
        re.search(
            r"不查询(?:具体)?企业事实|不查询外部数据|仅(?:保存|记录|写入)|"
            r"does not (?:query|retrieve|fetch)|write[- ]only|record[- ]only",
            description,
            re.I,
        )
    )


def link_checkpoints_to_evidence(
    checkpoints: Sequence[Mapping[str, Any]],
    risk_points: Sequence[Mapping[str, Any]],
    expectations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    points_by_label = {str(point.get("label") or ""): point for point in risk_points}
    valid_expectations = {str(item.get("label") or "") for item in expectations}
    linked = []
    for raw in checkpoints:
        checkpoint = dict(raw)
        declared = [
            item for item in _string_list(checkpoint.get("evidence_expectations")) if item in valid_expectations
        ]
        if not declared:
            point = points_by_label.get(str(checkpoint.get("label") or ""), {})
            checkpoint_text = " ".join(
                [
                    str(checkpoint.get("label") or ""),
                    " ".join(_string_list(checkpoint.get("keywords"))),
                    str(point.get("text") or ""),
                ]
            )
            scored = []
            for expectation in expectations:
                score = sum(
                    1
                    for keyword in _string_list(expectation.get("tool_intent_keywords"))
                    if keyword_matches(checkpoint_text, keyword)
                )
                if score:
                    scored.append((score, str(expectation.get("label") or "")))
            if scored:
                best_score = max(score for score, _label in scored)
                declared = sorted({label for score, label in scored if score == best_score and label})
        if declared:
            checkpoint["evidence_expectations"] = declared
            checkpoint["evidence_mode"] = "any" if str(checkpoint.get("evidence_mode")).lower() == "any" else "all"
        linked.append(checkpoint)
    return linked


def tool_coverage_metadata(
    checkpoints: Sequence[Mapping[str, Any]],
    expectations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize which expert checkpoints have an explicitly declared evidence tool."""
    supported_expectations = {
        str(expectation.get("label") or "") for expectation in expectations if trace_expectation_tool_names(expectation)
    }
    supported_count = 0
    linked_count = 0
    for checkpoint in checkpoints:
        required = _string_list(checkpoint.get("evidence_expectations"))
        if not required:
            continue
        linked_count += 1
        if str(checkpoint.get("evidence_mode") or "all").lower() == "any":
            supported = any(label in supported_expectations for label in required)
        else:
            supported = all(label in supported_expectations for label in required)
        supported_count += int(supported)

    checkpoint_count = len(checkpoints)
    if not checkpoint_count or not linked_count:
        coverage = "unmapped"
    elif supported_count == checkpoint_count:
        coverage = "complete"
    elif supported_count:
        coverage = "partial"
    else:
        coverage = "none"
    return {
        "tool_coverage": coverage,
        "tool_supported_checkpoint_count": supported_count,
        "linked_checkpoint_count": linked_count,
        "checkpoint_count": checkpoint_count,
    }


def trace_expectation_tool_names(expectation: Mapping[str, Any]) -> list[str]:
    return _string_list(expectation.get("tool_names"))


def keyword_matches(text: str, keyword: str) -> bool:
    compact_text = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(text)).casefold())
    compact_keyword = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(keyword)).casefold())
    return bool(compact_keyword) and compact_keyword in compact_text


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def is_action_only_checkpoint(label: str) -> bool:
    normalized = normalize_label(label)
    return bool(ACTION_ONLY_CHECKPOINT_PATTERN.search(normalized)) and not bool(
        RISK_SEMANTIC_PATTERN.search(normalized)
    )


if __name__ == "__main__":
    main()
