"""将信贷审批意见清洗为 GEPA JSONL 数据.

清洗器只提取审批官意见中的“项目风险点”章节. 智能体输入保持最小化, 通常
只有企业名称; 专家章节存入 ``data``, 仅供评估器和反思步骤使用.

示例:
    uv run --no-sync python examples/langchain_adapter/clean_credit_risk_dataset.py \
      --input-dir /path/to/opinions \
      --output examples/langchain_adapter/deepagents_gepa_credit_approval_project/evals/project_risk_sections.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

DEFAULT_RUBRIC = (
    "将评估材料视为审批官风险评价意见中的“项目风险点”章节. 评价智能体仅根据企业名称自主检索"
    "和分析后形成的轨迹与最终输出: 1. 是否通过成功的工具调用取得与企业特征相关的行业、财务、"
    "债务结构、集团穿透、环保安监、司法工商等证据; 2. 是否覆盖评估材料中的核心风险点; 3. 是否"
    "说明事实依据、比较方法和风险传导; 4. 是否识别缺失信息、工具能力缺口并提出补充核验; 5. 不"
    "奖励只套模板、未查证或针对单一样本硬编码的表述."
)

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
    parser.add_argument("--output", required=True, help="输出 JSONL 文件.")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC)
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
    records = []
    for index, source_file in enumerate(source_files, start=1):
        cleaned = clean_one_file(source_file, include_metadata=not args.no_metadata)
        if cleaned is None:
            continue
        record: dict[str, Any] = {
            "id": f"{args.id_prefix}_{index:04d}",
            "input": cleaned.company_name,
            "data": cleaned.section_text,
            "rubric": args.rubric,
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


def clean_one_file(source_file: Path, *, include_metadata: bool = True) -> CleanedRiskSection | None:
    text = read_source_text(source_file)
    extracted = extract_project_risk_section(text)
    if extracted is None:
        return None
    section_title, section_text = extracted
    company_name = infer_company_name(source_file, text)
    metadata = {
        "source_file": source_file.name,
        "section": section_title,
        "risk_points": extract_risk_points(section_text),
    }
    if include_metadata:
        metadata["checkpoints"] = build_checkpoints(metadata["risk_points"])
        metadata["trace_expectations"] = build_trace_expectations(metadata["risk_points"])
    return CleanedRiskSection(
        source_file=source_file,
        company_name=company_name,
        section_title=section_title,
        section_text=section_text,
        metadata=metadata,
    )


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
        from pypdf import PdfReader
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


def infer_company_name(path: Path, text: str) -> str:
    stem = path.stem
    stem = re.sub(r"(风险评价意见书|风险评价|授信审批|授信调查|尽调报告|项目风险点|审批意见)$", "", stem)
    stem = stem.strip(" _-")
    if stem:
        return stem
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9\uff08\uff09()]{2,80}(?:公司|集团|企业))", text)
    if match:
        return match.group(1)
    return path.stem


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
)


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


def build_checkpoints(risk_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = []
    for point in risk_points:
        label = str(point.get("label") or "项目风险点")
        keywords = [str(item) for item in point.get("keywords", []) if str(item).strip()]
        checkpoints.append({"label": label, "keywords": keywords or [label]})
    return checkpoints


def build_trace_expectations(risk_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: dict[str, list[str]] = {}
    joined_points = "\n".join(f"{point.get('label', '')}\n{point.get('text', '')}" for point in risk_points)
    compact = re.sub(r"\s+", "", joined_points)
    for label, keywords in KEYWORD_GROUPS:
        matched = [keyword for keyword in keywords if keyword in compact]
        if matched:
            labels[label] = matched
    return [
        {"label": label, "tool_intent_keywords": keywords}
        for label, keywords in sorted(labels.items(), key=lambda item: item[0])
    ]


if __name__ == "__main__":
    main()
