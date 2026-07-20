"""GEPA optimization of Deep Agents text surfaces.

This example demonstrates how to optimize the text artifacts that are passed to
``create_deep_agent`` while letting Deep Agents load memory and skills natively:

- ``AGENTS.md`` files passed through ``memory=[...]``
- the main agent ``system_prompt``
- main-agent tool descriptions
- subagent descriptions and system prompts
- subagent tool descriptions
- main-agent and subagent ``skills/*/SKILL.md`` files
- ``reference/*.md`` files inside those skill directories

It intentionally does not optimize ``scripts/*.py``, tool implementation code,
middleware code, or arbitrary source files. GEPA only sees named text
components; this example's discovery/apply helpers define how those components
map onto Deep Agents configuration.

Run:
    uv sync --extra dev --extra langchain
    uv pip install deepagents langchain-openai
    uv run python examples/langchain_adapter/deep_agent_skill_directory.py
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from examples.langchain_adapter.deepagents_gepa.artifacts import RunArtifactStore
    from examples.langchain_adapter.deepagents_gepa.framework import (
        ComponentSelector,
        Constraint,
        DatasetProvider,
        DefaultCandidateMaterializer,
        DefaultConstraintSet,
        DefaultDatasetProvider,
        DefaultEvaluator,
        DefaultFeedbackComponentSelector,
        DefaultProposalReviewer,
        DefaultReflectionTemplateRegistry,
        Evaluator,
        ProposalReviewer,
        ReflectionTemplateRegistry,
        select_deployment_candidate_index,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.langchain_adapter.deepagents_gepa.artifacts import RunArtifactStore
    from examples.langchain_adapter.deepagents_gepa.framework import (
        ComponentSelector,
        Constraint,
        DatasetProvider,
        DefaultCandidateMaterializer,
        DefaultConstraintSet,
        DefaultDatasetProvider,
        DefaultEvaluator,
        DefaultFeedbackComponentSelector,
        DefaultProposalReviewer,
        DefaultReflectionTemplateRegistry,
        Evaluator,
        ProposalReviewer,
        ReflectionTemplateRegistry,
        select_deployment_candidate_index,
    )

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback when tomli is installed.
    import tomli as tomllib  # type: ignore[no-redef]

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool

from gepa import optimize
from gepa.adapters.langchain_adapter import (
    LangChainAdapter,
    last_message_text,
    make_reflection_lm,
)

MAX_PROMPT_GROWTH = 0.50
MAX_COMPONENT_CHARS = {
    "memory": 12000,
    "prompt": 8000,
    "description": 1200,
    "skill": 18000,
    "reference": 12000,
}
DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000
DEFAULT_TRACE_CONTEXT_RATIO = 0.12
DEFAULT_TRACE_KEEP_RATIO = 0.10
DEFAULT_TRACE_CHARS_PER_TOKEN = 1.5
DEFAULT_TRACE_MIN_CHARS = 12_000
DEFAULT_TRACE_MAX_CHARS = 60_000
DEFAULT_TRACE_OMIT_TOOL_NAMES = frozenset({"edit_file", "write_file"})
RUNTIME_SPECIFIC_PATTERN = re.compile(
    r"(在 Claude Code|Claude Code skill|Claude Code 用户|Cursor only|Codex 中|"
    r"^\[!\[Claude Code|~/\.claude/skills/[a-z]|/plugin install\b)",
    re.I | re.M,
)
SOFTENER_PATTERN = re.compile(
    r"\b(consider|maybe|perhaps|flexibly|as appropriate|if possible|you may want to)\b",
    re.I,
)
ROUTE_RE = re.compile(r"<route>\s*([a-z_ -]+)\s*</route>", re.I)
SCRIPT_REFERENCE_RE = re.compile(r"(?P<path>(?:\./)?scripts/[A-Za-z0-9_./-]+\.py)")
COMPONENT_LABEL_RE = re.compile(r"(?m)^#{1,6}\s*(?:main:|memory:|skill:|subagent:|mcp:)")
YAML_FRONTMATTER_RE = re.compile(r"(?s)^\s*---\s*\n.*?\bname\s*:.*?\bdescription\s*:.*?\n---")
SKILL_DEFECT = "SKILL_DEFECT"
EXECUTION_LAPSE = "EXECUTION_LAPSE"
TOOL_CAPABILITY_GAP = "TOOL_CAPABILITY_GAP"
INSUFFICIENT_RUNTIME_EVIDENCE = "INSUFFICIENT_RUNTIME_EVIDENCE"
NO_FAILURE = "NO_FAILURE"
TOOL_FAILURE_PATTERN = re.compile(
    r"(?is)^\s*(?:error|exception|failed|failure|tool error|执行失败|调用失败|工具失败)\b|"
    r"\btraceback \(most recent call last\)"
)
LOGGER = logging.getLogger(__name__)


@dataclass
class ConstraintResult:
    passed: bool
    name: str
    message: str
    severity: str = "hard"


@dataclass
class ComponentSurface:
    key: str
    source_type: str
    relative_path: str | None = None
    source_path: str | None = None
    owner: str = "main"


@dataclass
class DeepAgentTextSpec:
    """Text surfaces discoverable from a create_deep_agent-style spec."""

    model: str | BaseChatModel | None
    root_dir: Path
    system_prompt: str
    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    subagents: list[dict[str, Any]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)


@dataclass
class CandidateApplication:
    kwargs: dict[str, Any]
    surfaces: dict[str, ComponentSurface]
    baseline_candidate: dict[str, str]
    temp_root: Path


@dataclass
class CapturedCreateDeepAgentCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class SurfaceConfig:
    """One explicitly declared text surface in a config-driven project."""

    name: str
    kind: str
    path: str | None = None
    component: str | None = None
    source_type: str | None = None
    owner: str = "main"
    include: tuple[str, ...] = ("SKILL.md", "reference/**/*.md")
    exclude: tuple[str, ...] = ("scripts/**",)


@dataclass(frozen=True)
class MCPServerConfig:
    """MCP server declaration carried through to the runtime layer."""

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPToolDescriptionConfig:
    """An MCP tool description that can be optimized as text."""

    name: str
    description: str
    server: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    source: str = "synthetic"
    path: str | None = None
    split: str = "train"
    limit: int | None = None
    query: dict[str, Any] = field(default_factory=dict)
    split_strategy: str = "stratified"
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    stratify_by: tuple[str, ...] = ("metadata.difficulty",)
    seed: int = 0
    evaluate_final_test: bool = True


@dataclass(frozen=True)
class DeepAgentsGepaConfig:
    """Config-first harness for different Deep Agents project shapes."""

    path: Path
    project_root: Path
    agent_mode: str = "manual"
    langgraph_config: str | None = None
    graph: str | None = None
    system_prompt: str = ""
    tools: tuple[str, ...] = ()
    memory: tuple[str, ...] = ("AGENTS.md",)
    skills: tuple[str, ...] = ("skills",)
    subagents: tuple[dict[str, Any], ...] = ()
    surfaces: tuple[SurfaceConfig, ...] = ()
    mcp_servers: tuple[MCPServerConfig, ...] = ()
    mcp_tool_descriptions: tuple[MCPToolDescriptionConfig, ...] = ()
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


@dataclass
class DeepAgentProjectCandidate:
    spec: DeepAgentTextSpec
    candidate: dict[str, str]
    surfaces: dict[str, ComponentSurface]
    config: DeepAgentsGepaConfig


@dataclass
class ConfiguredCandidateApplication:
    deep_agent_application: CandidateApplication
    config: DeepAgentsGepaConfig
    mcp_servers: tuple[MCPServerConfig, ...]
    mcp_tool_descriptions: dict[str, str]


@dataclass(frozen=True)
class EvalRecord:
    """Unified dataset row for golden, synthetic, and mined trace examples."""

    input: str | None = None
    data: str | None = None
    messages: tuple[dict[str, str], ...] = ()
    expected: str | None = None
    rubric: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_example(self) -> dict[str, Any]:
        text_input = self.input
        if text_input is None and self.messages:
            text_input = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}" for message in self.messages
            )
        example: dict[str, Any] = {
            "input": text_input or "",
            "rubric": self.rubric,
            "metadata": self.metadata,
        }
        if self.data is not None:
            example["data"] = self.data
        if self.messages:
            example["messages"] = [dict(message) for message in self.messages]
        if self.expected is not None:
            example["expected"] = self.expected
            example["answer"] = self.expected
        return example


DarwinFeedbackComponentSelector = DefaultFeedbackComponentSelector


def _expand_value(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _resolve_config_path(config_path: Path, raw_path: str | None, default: str = ".") -> Path:
    expanded = Path(_expand_value(raw_path or default))
    if not expanded.is_absolute():
        expanded = config_path.parent / expanded
    return expanded.resolve()


def _as_tuple(value: Any, default: Sequence[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def capture_create_deep_agent_calls():
    """Capture create_deep_agent kwargs while still returning the real graph."""
    try:
        import deepagents
        import deepagents.graph as deepagents_graph
    except ImportError as exc:
        raise ImportError("Install Deep Agents before loading langgraph_cli configs") from exc

    calls: list[CapturedCreateDeepAgentCall] = []
    original_package_fn = deepagents.create_deep_agent
    original_graph_fn = deepagents_graph.create_deep_agent

    def capture_wrapper(*args: Any, **kwargs: Any):
        calls.append(CapturedCreateDeepAgentCall(args=tuple(args), kwargs=dict(kwargs)))
        return original_graph_fn(*args, **kwargs)

    deepagents.create_deep_agent = capture_wrapper
    deepagents_graph.create_deep_agent = capture_wrapper
    try:
        yield calls
    finally:
        deepagents.create_deep_agent = original_package_fn
        deepagents_graph.create_deep_agent = original_graph_fn


def load_deepagents_gepa_config(path: str | Path) -> DeepAgentsGepaConfig:
    """Load a config-driven Deep Agents GEPA harness.

    The config is intentionally declarative. It says where runtime text lives
    and how to run the agent; it does not ask GEPA to infer arbitrary Python
    closures or rewrite source code.
    """
    config_path = Path(path).expanduser().resolve()
    payload = _load_toml(config_path)
    experiment = payload.get("experiment", {})
    agent = payload.get("agent", {})
    project_root = _resolve_config_path(
        config_path,
        str(agent.get("project_root") or experiment.get("project_root") or experiment.get("workspace_root") or "."),
    )
    return DeepAgentsGepaConfig(
        path=config_path,
        project_root=project_root,
        agent_mode=_normalize_agent_mode(str(agent.get("mode", "manual"))),
        langgraph_config=agent.get("langgraph_config"),
        graph=agent.get("graph"),
        system_prompt=str(agent.get("system_prompt", "")),
        tools=_as_tuple(agent.get("tools")),
        memory=_as_tuple(agent.get("memory"), ("AGENTS.md",)),
        skills=_as_tuple(agent.get("skills"), ("skills",)),
        subagents=tuple(dict(item) for item in agent.get("subagents", [])),
        surfaces=_parse_surface_configs(payload.get("surfaces", {})),
        mcp_servers=_parse_mcp_servers(payload.get("mcp", {})),
        mcp_tool_descriptions=_parse_mcp_tool_descriptions(payload.get("mcp", {})),
        dataset=_parse_dataset_config(payload.get("dataset", {})),
    )


def _normalize_agent_mode(mode: str) -> str:
    if mode == "filesystem":
        return "manual"
    return mode


def _parse_surface_configs(raw_surfaces: dict[str, Any]) -> tuple[SurfaceConfig, ...]:
    surfaces: list[SurfaceConfig] = []
    for name, raw in raw_surfaces.items():
        if not isinstance(raw, dict):
            continue
        surfaces.append(
            SurfaceConfig(
                name=str(name),
                kind=str(raw.get("kind", "file")),
                path=raw.get("path") or raw.get("target"),
                component=raw.get("component"),
                source_type=raw.get("source_type"),
                owner=str(raw.get("owner", "main")),
                include=_as_tuple(raw.get("include"), ("SKILL.md", "reference/**/*.md")),
                exclude=_as_tuple(raw.get("exclude"), ("scripts/**",)),
            )
        )
    return tuple(surfaces)


def _parse_mcp_servers(raw_mcp: dict[str, Any]) -> tuple[MCPServerConfig, ...]:
    servers: list[MCPServerConfig] = []
    for item in raw_mcp.get("servers", []):
        if not isinstance(item, dict):
            continue
        servers.append(
            MCPServerConfig(
                name=str(item["name"]),
                transport=str(item.get("transport", "stdio")),
                command=item.get("command"),
                args=_as_tuple(item.get("args")),
                url=item.get("url"),
                env={str(key): str(value) for key, value in item.get("env", {}).items()},
            )
        )
    return tuple(servers)


def _parse_mcp_tool_descriptions(raw_mcp: dict[str, Any]) -> tuple[MCPToolDescriptionConfig, ...]:
    tools: list[MCPToolDescriptionConfig] = []
    for item in raw_mcp.get("tools", []):
        if not isinstance(item, dict):
            continue
        tools.append(
            MCPToolDescriptionConfig(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                server=item.get("server"),
            )
        )
    return tuple(tools)


def _parse_dataset_config(raw_dataset: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        source=str(raw_dataset.get("source", "synthetic")),
        path=raw_dataset.get("path"),
        split=str(raw_dataset.get("split", "train")),
        limit=int(raw_dataset["limit"]) if raw_dataset.get("limit") is not None else None,
        query=dict(raw_dataset.get("query", {})),
        split_strategy=str(raw_dataset.get("split_strategy", "stratified")),
        train_ratio=float(raw_dataset.get("train_ratio", 0.60)),
        val_ratio=float(raw_dataset.get("val_ratio", 0.20)),
        test_ratio=float(raw_dataset.get("test_ratio", 0.20)),
        stratify_by=_as_tuple(raw_dataset.get("stratify_by"), ("metadata.difficulty",)),
        seed=int(raw_dataset.get("seed", 0)),
        evaluate_final_test=bool(raw_dataset.get("evaluate_final_test", True)),
    )


def _import_from_ref(ref: str, project_root: Path) -> Any:
    if ":" not in ref:
        raise ValueError(f"Import reference must use module:attribute syntax: {ref}")
    module_name, attr_path = ref.split(":", 1)
    module_path = Path(module_name)
    if module_name.endswith(".py") or module_name.startswith("."):
        if not module_path.is_absolute():
            module_path = project_root / module_path
        inserted = False
        root_string = str(project_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
            inserted = True
        spec = importlib.util.spec_from_file_location(f"_gepa_dynamic_{abs(hash(module_path))}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            obj: Any = module
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return obj
        finally:
            if inserted:
                try:
                    sys.path.remove(root_string)
                except ValueError:
                    pass
    inserted = False
    root_string = str(project_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
        inserted = True
    try:
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj
    finally:
        if inserted:
            try:
                sys.path.remove(root_string)
            except ValueError:
                pass


def build_deep_agent_spec_from_config(
    config: DeepAgentsGepaConfig,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None = None,
) -> DeepAgentTextSpec:
    """Build a create_deep_agent-like spec from config.

    Manual mode reads explicit TOML fields. LangGraph CLI mode reads the graph
    entry from langgraph.json and expects that graph entry to expose
    DeepAgentTextSpec or create_deep_agent-style kwargs.
    """
    if config.agent_mode == "langgraph_cli":
        return build_deep_agent_spec_from_langgraph_config(config, tool_registry)
    return _spec_from_kwargs(
        config,
        {
            "system_prompt": config.system_prompt,
            "tools": list(config.tools),
            "subagents": list(config.subagents),
            "skills": list(config.skills),
            "memory": list(config.memory),
        },
        tool_registry,
    )


def build_deep_agent_spec_from_langgraph_config(
    config: DeepAgentsGepaConfig,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None,
) -> DeepAgentTextSpec:
    langgraph_config = _resolve_source_path(config.project_root, config.langgraph_config or "langgraph.json")
    payload = _load_json(langgraph_config)
    graphs = payload.get("graphs", {})
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError(f"{langgraph_config} must contain a non-empty graphs object")
    graph_name = config.graph or next(iter(graphs))
    if graph_name not in graphs:
        raise ValueError(f"Graph {graph_name!r} not found in {langgraph_config}")
    graph_ref = str(graphs[graph_name])
    with capture_create_deep_agent_calls() as captured_calls:
        graph_obj = _import_from_ref(graph_ref, config.project_root)
        if callable(graph_obj):
            call_langgraph_factory(graph_obj)
    if captured_calls:
        return spec_from_create_deep_agent_call(captured_calls[-1], config, tool_registry)
    raise TypeError(
        "LangGraph graph entry did not call deepagents.create_deep_agent. "
        "GEPA can auto-discover only create_deep_agent-based graphs."
    )


def call_langgraph_factory(graph_obj: Callable[..., Any]) -> Any:
    """Call a LangGraph CLI graph factory with RunnableConfig-shaped input."""
    try:
        return graph_obj({})
    except TypeError:
        return graph_obj()


def coerce_deep_agent_spec(
    value: Any,
    config: DeepAgentsGepaConfig,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None,
) -> DeepAgentTextSpec | None:
    if isinstance(value, DeepAgentTextSpec):
        return value
    if isinstance(value, dict):
        return _spec_from_kwargs(config, value, tool_registry)
    return None


def spec_from_create_deep_agent_call(
    call: CapturedCreateDeepAgentCall,
    config: DeepAgentsGepaConfig,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None,
) -> DeepAgentTextSpec:
    kwargs = dict(call.kwargs)
    if call.args:
        kwargs.setdefault("model", call.args[0])
    if len(call.args) > 1:
        kwargs.setdefault("tools", call.args[1])
    return _spec_from_kwargs(config, kwargs, tool_registry)


def _spec_from_kwargs(
    config: DeepAgentsGepaConfig,
    kwargs: dict[str, Any],
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None,
) -> DeepAgentTextSpec:
    tools = [_resolve_tool_ref(item, config.project_root, tool_registry) for item in kwargs.get("tools", [])]
    subagents = []
    for subagent in kwargs.get("subagents", []):
        copied = dict(subagent)
        copied["tools"] = [
            _resolve_tool_ref(item, config.project_root, tool_registry) for item in copied.get("tools", [])
        ]
        copied["skills"] = list(copied.get("skills", []))
        subagents.append(copied)
    return DeepAgentTextSpec(
        model=kwargs.get("model"),
        root_dir=config.project_root,
        system_prompt=str(kwargs.get("system_prompt", "")),
        tools=tools,
        subagents=subagents,
        skills=list(kwargs.get("skills", [])),
        memory=list(kwargs.get("memory", [])),
    )


def _resolve_tool_ref(
    value: Any,
    project_root: Path,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None,
) -> BaseTool | Callable | dict[str, Any]:
    if not isinstance(value, str):
        return value
    if tool_registry and value in tool_registry:
        return tool_registry[value]
    return _import_from_ref(value, project_root)


def build_candidate_from_deep_agent_project(
    config: DeepAgentsGepaConfig,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None = None,
) -> DeepAgentProjectCandidate:
    spec = build_deep_agent_spec_from_config(config, tool_registry)
    candidate, surfaces = build_candidate_from_deep_agent_spec(spec)
    _add_config_surfaces(candidate, surfaces, config)
    _add_mcp_tool_description_surfaces(candidate, surfaces, config)
    return DeepAgentProjectCandidate(spec=spec, candidate=candidate, surfaces=surfaces, config=config)


def _add_config_surfaces(
    candidate: dict[str, str],
    surfaces: dict[str, ComponentSurface],
    config: DeepAgentsGepaConfig,
) -> None:
    for surface in config.surfaces:
        if surface.kind == "file":
            if surface.path is None:
                raise ValueError(f"surface {surface.name} must define path")
            path = _resolve_source_path(config.project_root, surface.path)
            key = surface.component or surface.name
            candidate[key] = _read_text(path)
            surfaces[key] = ComponentSurface(
                key,
                surface.source_type or _infer_surface_type(key, path),
                _posix_path(path.relative_to(config.project_root)),
                owner=surface.owner,
            )
        elif surface.kind == "skill_dir":
            if surface.path is None:
                raise ValueError(f"surface {surface.name} must define path")
            source_path = _source_relative_to_root(config.project_root, surface.path)
            for rel_id, file_path in _iter_skill_files(config.project_root, surface.path):
                prefix = f"subagent:{surface.owner}:skill" if surface.owner != "main" else "skill"
                key = surface.component or f"{prefix}:{rel_id}"
                candidate[key] = _read_text(file_path)
                surfaces[key] = ComponentSurface(
                    key,
                    _surface_type_for_skill_file(rel_id),
                    _posix_path(file_path.relative_to(config.project_root)),
                    source_path=source_path,
                    owner=surface.owner,
                )


def _infer_surface_type(key: str, path: Path) -> str:
    if path.name == "AGENTS.md" or key.startswith("memory:"):
        return "memory"
    if path.name == "SKILL.md":
        return "skill"
    if "reference" in path.parts:
        return "reference"
    if key.endswith(":description"):
        return "description"
    return "prompt"


def _add_mcp_tool_description_surfaces(
    candidate: dict[str, str],
    surfaces: dict[str, ComponentSurface],
    config: DeepAgentsGepaConfig,
) -> None:
    for tool_config in config.mcp_tool_descriptions:
        key = f"mcp:tool:{tool_config.name}:description"
        candidate[key] = tool_config.description
        surfaces[key] = ComponentSurface(key, "description", owner=tool_config.server or "mcp")


def apply_candidate_to_deep_agent_project(
    project: DeepAgentProjectCandidate,
    candidate: dict[str, str],
    temp_root: Path,
) -> ConfiguredCandidateApplication:
    copy_config_skill_sources_to_temp(project, temp_root)
    application = apply_candidate_to_deep_agent_spec(project.spec, candidate, project.surfaces, temp_root)
    for key, surface in project.surfaces.items():
        if surface.relative_path is None:
            continue
        if key.startswith(("memory:", "skill:", "subagent:")) and ":description" not in key:
            continue
        if key.startswith("mcp:"):
            continue
        _write_text(temp_root / surface.relative_path, candidate[key])
    mcp_descriptions = {
        tool.name: candidate.get(f"mcp:tool:{tool.name}:description", tool.description)
        for tool in project.config.mcp_tool_descriptions
    }
    return ConfiguredCandidateApplication(
        deep_agent_application=application,
        config=project.config,
        mcp_servers=project.config.mcp_servers,
        mcp_tool_descriptions=mcp_descriptions,
    )


def copy_config_skill_sources_to_temp(project: DeepAgentProjectCandidate, temp_root: Path) -> None:
    """Mirror skill_dir surfaces declared only in TOML before writing candidates."""
    copied: set[str] = set()
    for surface in project.surfaces.values():
        if surface.source_path is None or surface.source_path in copied:
            continue
        source_dir = _resolve_source_path(project.config.project_root, surface.source_path)
        if not source_dir.exists():
            continue
        out_dir = temp_root / surface.source_path
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(source_dir, out_dir)
        copied.add(surface.source_path)


def _posix_path(path: str | Path) -> str:
    return str(PurePosixPath(str(path).replace("\\", "/")))


def _resolve_source_path(root_dir: Path, source_path: str) -> Path:
    path = Path(source_path)
    if not path.is_absolute():
        path = root_dir / source_path
    return path.resolve()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tool_name(tool_obj: BaseTool | Callable | dict[str, Any]) -> str:
    if isinstance(tool_obj, dict):
        return str(tool_obj.get("name") or tool_obj.get("function", {}).get("name") or "unnamed_tool")
    return str(getattr(tool_obj, "name", None) or getattr(tool_obj, "__name__", "unnamed_tool"))


def _tool_description(tool_obj: BaseTool | Callable | dict[str, Any]) -> str:
    if isinstance(tool_obj, dict):
        return str(tool_obj.get("description") or tool_obj.get("function", {}).get("description") or "")
    return str(getattr(tool_obj, "description", None) or getattr(tool_obj, "__doc__", "") or "").strip()


def tool_inventory_from_kwargs(kwargs: Mapping[str, Any]) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for tool_obj in kwargs.get("tools", []) or []:
        inventory.append(
            {
                "owner": "main",
                "name": _tool_name(tool_obj),
                "description": _tool_description(tool_obj),
            }
        )
    for subagent in kwargs.get("subagents", []) or []:
        if not isinstance(subagent, Mapping):
            continue
        owner = str(subagent.get("name") or "subagent")
        for tool_obj in subagent.get("tools", []) or []:
            inventory.append(
                {
                    "owner": owner,
                    "name": _tool_name(tool_obj),
                    "description": _tool_description(tool_obj),
                }
            )
    return inventory


def tool_inventory_text(inventory: Sequence[Mapping[str, str]]) -> str:
    lines = []
    for item in inventory:
        lines.append(f"{item.get('owner', 'main')}::{item.get('name', '')}: {item.get('description', '')}")
    return "\n".join(lines)


def stable_tool_inventory(
    runtime_inventory: Sequence[Mapping[str, str]],
    baseline_candidate: Mapping[str, str],
) -> list[dict[str, str]]:
    """Use seed descriptions for capability diagnosis.

    Tool descriptions are optimizable text. Capability-gap detection must not
    let a proposal claim that an unchanged tool implementation gained a new
    data source merely by rewriting its description.
    """
    stable: list[dict[str, str]] = []
    for raw_item in runtime_inventory:
        item = {str(key): str(value) for key, value in raw_item.items()}
        owner = item.get("owner", "main")
        name = item.get("name", "")
        candidate_keys = [
            f"main:tool:{name}:description" if owner == "main" else f"subagent:{owner}:tool:{name}:description",
            f"mcp:tool:{name}:description",
        ]
        for key in candidate_keys:
            if key in baseline_candidate:
                item["description"] = str(baseline_candidate[key])
                break
        stable.append(item)
    return stable


def _copy_tool_with_description(tool_obj: BaseTool | Callable | dict[str, Any], description: str):
    if isinstance(tool_obj, dict):
        copied = copy.deepcopy(tool_obj)
        if "function" in copied and isinstance(copied["function"], dict):
            copied["function"]["description"] = description
        else:
            copied["description"] = description
        return copied
    if isinstance(tool_obj, BaseTool):
        if hasattr(tool_obj, "model_copy"):
            return tool_obj.model_copy(update={"description": description})
        if hasattr(tool_obj, "copy"):
            return tool_obj.copy(update={"description": description})
        copied = copy.copy(tool_obj)
        copied.description = description
        return copied
    if callable(tool_obj):
        return StructuredTool.from_function(func=tool_obj, description=description)
    return tool_obj


def _iter_skill_files(root_dir: Path, source: str) -> list[tuple[str, Path]]:
    source_dir = _resolve_source_path(root_dir, source)
    if not source_dir.exists():
        return []
    files: list[tuple[str, Path]] = []
    for skill_md in sorted(source_dir.glob("*/SKILL.md")):
        skill_dir = skill_md.parent
        skill_name = skill_dir.name
        files.append((f"{skill_name}:SKILL.md", skill_md))
        reference_dir = skill_dir / "reference"
        if reference_dir.exists():
            for ref_file in sorted(reference_dir.rglob("*")):
                if ref_file.is_file():
                    rel = ref_file.relative_to(skill_dir).as_posix()
                    files.append((f"{skill_name}:{rel}", ref_file))
    return files


def _source_relative_to_root(root_dir: Path, source: str) -> str:
    source_dir = _resolve_source_path(root_dir, source)
    try:
        return _posix_path(source_dir.relative_to(root_dir))
    except ValueError:
        return _posix_path(source)


def build_candidate_from_deep_agent_spec(spec: DeepAgentTextSpec) -> tuple[dict[str, str], dict[str, ComponentSurface]]:
    """Discover optimizable text surfaces from explicit Deep Agents inputs.

    Discovery is intentionally conservative: only explicit arguments and files
    referenced by those arguments become candidate components.
    """
    candidate: dict[str, str] = {}
    surfaces: dict[str, ComponentSurface] = {}

    candidate["main:system_prompt"] = spec.system_prompt
    surfaces["main:system_prompt"] = ComponentSurface("main:system_prompt", "prompt")

    for memory_path in spec.memory:
        resolved = _resolve_source_path(spec.root_dir, memory_path)
        key = f"memory:{Path(memory_path).name}"
        candidate[key] = _read_text(resolved)
        surfaces[key] = ComponentSurface(key, "memory", _posix_path(memory_path))

    for tool_obj in spec.tools:
        tool_name = _tool_name(tool_obj)
        key = f"main:tool:{tool_name}:description"
        candidate[key] = _tool_description(tool_obj)
        surfaces[key] = ComponentSurface(key, "description")

    for source in spec.skills:
        source_path = _source_relative_to_root(spec.root_dir, source)
        for rel_id, file_path in _iter_skill_files(spec.root_dir, source):
            key = f"skill:{rel_id}"
            candidate[key] = _read_text(file_path)
            surfaces[key] = ComponentSurface(
                key,
                _surface_type_for_skill_file(rel_id),
                _posix_path(file_path.relative_to(spec.root_dir)),
                source_path=source_path,
            )

    for subagent in spec.subagents:
        name = str(subagent["name"])
        desc_key = f"subagent:{name}:description"
        prompt_key = f"subagent:{name}:system_prompt"
        candidate[desc_key] = str(subagent["description"])
        candidate[prompt_key] = str(subagent["system_prompt"])
        surfaces[desc_key] = ComponentSurface(desc_key, "description", owner=name)
        surfaces[prompt_key] = ComponentSurface(prompt_key, "prompt", owner=name)

        for tool_obj in subagent.get("tools", []):
            tool_name = _tool_name(tool_obj)
            key = f"subagent:{name}:tool:{tool_name}:description"
            candidate[key] = _tool_description(tool_obj)
            surfaces[key] = ComponentSurface(key, "description", owner=name)

        for source in subagent.get("skills", []):
            source_path = _source_relative_to_root(spec.root_dir, source)
            for rel_id, file_path in _iter_skill_files(spec.root_dir, source):
                key = f"subagent:{name}:skill:{rel_id}"
                candidate[key] = _read_text(file_path)
                surfaces[key] = ComponentSurface(
                    key,
                    _surface_type_for_skill_file(rel_id),
                    _posix_path(file_path.relative_to(spec.root_dir)),
                    source_path=source_path,
                    owner=name,
                )

    return candidate, surfaces


def _surface_type_for_skill_file(rel_id: str) -> str:
    return "skill" if rel_id.endswith(":SKILL.md") else "reference"


def apply_candidate_to_deep_agent_spec(
    spec: DeepAgentTextSpec,
    candidate: dict[str, str],
    surfaces: dict[str, ComponentSurface],
    temp_root: Path,
) -> CandidateApplication:
    """Write a candidate to a temp Deep Agents workspace and return kwargs."""
    baseline_candidate, _ = build_candidate_from_deep_agent_spec(spec)
    memory_paths: list[str] = []
    skill_sources: list[str] = []
    subagent_skill_sources: dict[str, list[str]] = {}

    copy_static_sources_to_temp(spec, temp_root)

    for key, surface in surfaces.items():
        text = candidate[key]
        if key.startswith("memory:"):
            out = temp_root / (surface.relative_path or "AGENTS.md")
            _write_text(out, text)
            memory_paths.append(_posix_path(out.relative_to(temp_root)))
        elif key.startswith("skill:"):
            rel = surface.relative_path
            if rel is None:
                continue
            out = temp_root / rel
            _write_text(out, text)
            source_path = surface.source_path or _posix_path(Path(rel).parts[0])
            if source_path not in skill_sources:
                skill_sources.append(source_path)
        elif key.startswith("subagent:") and ":skill:" in key:
            rel = surface.relative_path
            if rel is None:
                continue
            out = temp_root / rel
            _write_text(out, text)
            source_path = surface.source_path or _posix_path(Path(rel).parts[0])
            subagent_skill_sources.setdefault(surface.owner, [])
            if source_path not in subagent_skill_sources[surface.owner]:
                subagent_skill_sources[surface.owner].append(source_path)

    materialize_referenced_skill_script_aliases(candidate, surfaces, temp_root)

    main_tools = []
    for tool_obj in spec.tools:
        tool_name = _tool_name(tool_obj)
        key = f"main:tool:{tool_name}:description"
        main_tools.append(_copy_tool_with_description(tool_obj, candidate.get(key, _tool_description(tool_obj))))

    subagents = []
    for original in spec.subagents:
        name = str(original["name"])
        copied = dict(original)
        copied["description"] = candidate.get(f"subagent:{name}:description", str(original["description"]))
        copied["system_prompt"] = candidate.get(f"subagent:{name}:system_prompt", str(original["system_prompt"]))
        copied["skills"] = subagent_skill_sources.get(name, list(original.get("skills", [])))
        if "tools" in original:
            copied_tools = []
            for tool_obj in original["tools"]:
                tool_name = _tool_name(tool_obj)
                key = f"subagent:{name}:tool:{tool_name}:description"
                copied_tools.append(
                    _copy_tool_with_description(tool_obj, candidate.get(key, _tool_description(tool_obj)))
                )
            copied["tools"] = copied_tools
        subagents.append(copied)

    kwargs = {
        "model": spec.model,
        "system_prompt": candidate["main:system_prompt"],
        "tools": main_tools,
        "subagents": subagents,
        "skills": skill_sources,
        "memory": memory_paths,
    }
    return CandidateApplication(kwargs, surfaces, baseline_candidate, temp_root)


def copy_static_sources_to_temp(spec: DeepAgentTextSpec, temp_root: Path) -> None:
    """Copy non-candidate files such as skill scripts into the temp workspace.

    Skill source directories are mirrored into the temp tree before candidate
    text is written. Removing the destination first avoids stale reference or
    script files from a previous materialization surviving by accident.
    """
    for source in [*spec.skills, *(s for subagent in spec.subagents for s in subagent.get("skills", []))]:
        source_dir = _resolve_source_path(spec.root_dir, source)
        if not source_dir.exists():
            continue
        rel_source = _source_relative_to_root(spec.root_dir, source)
        out_dir = temp_root / rel_source
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(source_dir, out_dir)


def materialize_referenced_skill_script_aliases(
    candidate: dict[str, str],
    surfaces: dict[str, ComponentSurface],
    temp_root: Path,
) -> None:
    """Expose referenced skill scripts for `python scripts/foo.py` workflows.

    Deep Agents asks agents to use absolute paths from the skill list, but many
    existing skills contain relative commands such as `python scripts/build.py`.
    The scripts remain non-optimized static files; this helper only creates
    root-level aliases in the temporary workspace when the referenced script is
    present under the owning skill directory.
    """
    alias_root = temp_root / "scripts"
    if alias_root.exists():
        shutil.rmtree(alias_root)
    for key, text in candidate.items():
        surface = surfaces.get(key)
        if surface is None or surface.source_type != "skill" or surface.relative_path is None:
            continue
        skill_dir = temp_root / surface.relative_path
        skill_dir = skill_dir.parent
        for script_ref in referenced_script_paths(text):
            source = skill_dir / script_ref
            if not source.exists():
                continue
            destination = temp_root / script_ref
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def referenced_script_paths(text: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for match in SCRIPT_REFERENCE_RE.finditer(text):
        raw_path = match.group("path").removeprefix("./")
        if ".." in Path(raw_path).parts or raw_path in seen:
            continue
        paths.append(Path(raw_path))
        seen.add(raw_path)
    return paths


def reflection_prompt_templates(candidate: dict[str, str]) -> dict[str, str]:
    return DefaultReflectionTemplateRegistry().templates_for(candidate)


@tool
def tag_ticket(ticket: str, route: str) -> str:
    """Tag a support ticket with route billing, account, engineering, or product."""
    return f"{ticket} -> {route}"


@tool
def lookup_policy(topic: str) -> str:
    """Look up support policy notes for a topic."""
    policies = {
        "billing": "Billing includes invoices, refunds, duplicate charges, receipts, and plan changes.",
        "account": "Account includes login, password, authentication, locked access, and profile ownership.",
        "engineering": "Engineering includes bugs, crashes, errors, regressions, and broken features.",
        "product": "Product includes feature requests, roadmap, integrations, and workflow improvements.",
    }
    return policies.get(topic.lower(), "No policy found.")


def create_seed_workspace(root: Path) -> DeepAgentTextSpec:
    _write_text(
        root / "AGENTS.md",
        """# Support Router Agent

You route support requests to one team. Be precise, concise, and use the support-router skill before finalizing.

## Output

Return the final route inside <route>...</route> tags.
""",
    )
    _write_text(
        root / "skills" / "support-router" / "SKILL.md",
        """---
name: support-router
description: Route support requests to billing, account, engineering, or product.
---

# Support Router

Use this skill when a support request needs a destination team.

## Workflow

1. Read `reference/routing.md`.
2. Identify the main user intent.
3. If routing evidence is ambiguous, run `python scripts/ignored.py` to inspect helper output.
4. Use the final output contract from `reference/output.md`.

## Failure Modes and Guardrails

- If a request mentions money, invoices, charges, receipts, or refunds, do not route it to product.
- If a request is about login, password, authentication, or locked access, do not route it to billing.
- If the request reports a crash, error, or broken feature, do not route it to account.
""",
    )
    _write_text(
        root / "skills" / "support-router" / "reference" / "routing.md",
        """# Routing Policy

- Billing, invoices, refunds, charges, receipts -> billing
- Login, password, account access, authentication -> account
- Bugs, errors, crashes, broken features -> engineering
- Feature requests, integrations, roadmap questions -> product
""",
    )
    _write_text(
        root / "skills" / "support-router" / "reference" / "output.md",
        """# Output Contract

Return exactly one tag: `<route>TEAM</route>`.

TEAM must be one of: billing, account, engineering, product.
""",
    )
    _write_text(
        root / "skills" / "support-router" / "scripts" / "ignored.py",
        "SHOULD_NOT_BE_OPTIMIZED = True\n",
    )
    _write_text(
        root / "subagents" / "triage" / "skills" / "triage-notes" / "SKILL.md",
        """---
name: triage-notes
description: Summarize which routing evidence matters.
---

# Triage Notes

Use this skill when the main agent delegates ambiguous routing work.

## Workflow

1. Extract the strongest routing evidence.
2. If extra evidence is needed, run `python scripts/ignored.py`.
3. List one likely team and one rejected alternative.

## Guardrails

- Do not invent user details.
- If evidence is weak, say what is missing.
""",
    )
    _write_text(
        root / "subagents" / "triage" / "skills" / "triage-notes" / "reference" / "signals.md",
        """# Routing Signals

- Money words usually indicate billing.
- Access words usually indicate account.
- Crash or error words usually indicate engineering.
- Roadmap or integration words usually indicate product.
""",
    )
    _write_text(
        root / "subagents" / "triage" / "skills" / "triage-notes" / "scripts" / "ignored.py",
        "SHOULD_NOT_BE_OPTIMIZED = True\n",
    )
    subagents = [
        {
            "name": "triage",
            "description": "Use for ambiguous support routing before choosing a team.",
            "system_prompt": "You are a triage assistant. Use lookup_policy and the triage-notes skill.",
            "tools": [lookup_policy],
            "skills": ["subagents/triage/skills"],
        }
    ]
    return DeepAgentTextSpec(
        model=None,
        root_dir=root,
        system_prompt="You are a support router. Use available memory, skills, and tools before answering.",
        tools=[tag_ticket],
        subagents=subagents,
        skills=["skills"],
        memory=["AGENTS.md"],
    )


def create_deep_agent_from_application(application: CandidateApplication, model: BaseChatModel):
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise ImportError("Install Deep Agents before running this example: uv pip install deepagents") from exc

    kwargs = dict(application.kwargs)
    kwargs["model"] = model
    kwargs["backend"] = create_executable_deep_agent_backend(application.temp_root)
    return create_deep_agent(**kwargs)


def create_executable_deep_agent_backend(root_dir: Path):
    """Create the backend used by this example's Deep Agents runtime.

    Skill scripts are runtime resources, not GEPA candidate text. Using
    LocalShellBackend lets Deep Agents' `execute` tool run those scripts from
    the temporary candidate workspace.
    """
    try:
        from deepagents.backends import LocalShellBackend
    except ImportError as exc:
        raise ImportError("Install Deep Agents before running this example: uv pip install deepagents") from exc
    return LocalShellBackend(root_dir=root_dir, virtual_mode=True, inherit_env=True)


def create_deep_agent_from_configured_application(
    application: ConfiguredCandidateApplication,
    model: BaseChatModel,
    mcp_loader: Callable[[Sequence[MCPServerConfig], dict[str, str]], Sequence[BaseTool | Callable | dict[str, Any]]]
    | None = None,
):
    deep_agent_application = configured_runtime_application(application, mcp_loader)
    return create_deep_agent_from_application(deep_agent_application, model)


def configured_runtime_application(
    application: ConfiguredCandidateApplication,
    mcp_loader: Callable[[Sequence[MCPServerConfig], dict[str, str]], Sequence[BaseTool | Callable | dict[str, Any]]]
    | None = None,
) -> CandidateApplication:
    deep_agent_application = application.deep_agent_application
    if mcp_loader is not None and application.mcp_servers:
        kwargs = dict(deep_agent_application.kwargs)
        kwargs["tools"] = list(kwargs.get("tools", [])) + list(
            mcp_loader(application.mcp_servers, application.mcp_tool_descriptions)
        )
        deep_agent_application = CandidateApplication(
            kwargs=kwargs,
            surfaces=deep_agent_application.surfaces,
            baseline_candidate=deep_agent_application.baseline_candidate,
            temp_root=deep_agent_application.temp_root,
        )
    return deep_agent_application


def rollout(
    candidate: dict[str, str],
    example: dict[str, Any],
    llm: BaseChatModel,
    seed_spec: DeepAgentTextSpec,
    surfaces: dict[str, ComponentSurface],
    baseline_candidate: dict[str, str],
) -> dict:
    with tempfile.TemporaryDirectory(prefix="gepa_deep_agent_text_surfaces_") as tmp:
        temp_root = Path(tmp)
        materializer = DefaultCandidateMaterializer(
            lambda current_candidate, output_dir: apply_candidate_to_deep_agent_spec(
                seed_spec,
                dict(current_candidate),
                surfaces,
                output_dir,
            )
        )
        application = materializer.materialize(candidate, temp_root)
        constraints = DefaultConstraintSet(
            validate_candidate_constraints,
            baseline_candidate,
            surfaces,
        ).check(candidate, {"materialized_root": temp_root})
        runtime_inventory = tool_inventory_from_kwargs(application.kwargs)
        state_extras = {
            "available_tools": runtime_inventory,
            "capability_tools": stable_tool_inventory(runtime_inventory, baseline_candidate),
            "trace_context_window_tokens": trace_context_window_tokens(),
            "trace_context_ratio": trace_context_ratio(),
            "evaluation_phase": example.get("evaluation_phase", "optimization"),
        }
        critical_failures = critical_constraint_results(constraints)
        if critical_failures:
            state = {
                "messages": [],
                "candidate_runtime_skipped": True,
                "candidate_runtime_skip_reason": critical_failure_summary(critical_failures),
            }
        else:
            try:
                agent = create_deep_agent_from_application(application, llm)
                state = agent.invoke({"messages": [HumanMessage(content=example["input"])]})
            except Exception as exc:
                state = {"messages": [], "error": exc}
        if not isinstance(state, dict):
            state = {"messages": getattr(state, "messages", [])}
        state.update(state_extras)
        state["candidate_hash"] = candidate_hash(candidate)
        state["candidate_excerpt"] = summarize_candidate(candidate)
        state["candidate_constraints"] = [constraint.__dict__ for constraint in constraints]
        state["candidate_metrics"] = candidate_metrics(candidate, baseline_candidate)
        state["baseline_response"] = (
            "" if critical_failures else run_baseline_for_example(example, llm, seed_spec, surfaces, baseline_candidate)
        )
        return state


def configured_rollout(
    candidate: dict[str, str],
    example: dict[str, Any],
    llm: BaseChatModel,
    project: DeepAgentProjectCandidate,
    baseline_candidate: dict[str, str],
    mcp_loader: Callable[[Sequence[MCPServerConfig], dict[str, str]], Sequence[BaseTool | Callable | dict[str, Any]]]
    | None = None,
    constraint_policy: Constraint | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="gepa_deep_agent_project_") as tmp:
        temp_root = Path(tmp)
        materializer = DefaultCandidateMaterializer(
            lambda current_candidate, output_dir: apply_candidate_to_deep_agent_project(
                project,
                dict(current_candidate),
                output_dir,
            )
        )
        application = materializer.materialize(candidate, temp_root)
        constraints = (
            constraint_policy
            or DefaultConstraintSet(
                validate_candidate_constraints,
                baseline_candidate,
                project.surfaces,
            )
        ).check(candidate, {"materialized_root": temp_root})
        critical_failures = critical_constraint_results(constraints)
        runtime_application = (
            application.deep_agent_application
            if critical_failures
            else configured_runtime_application(application, mcp_loader)
        )
        runtime_inventory = tool_inventory_from_kwargs(runtime_application.kwargs)
        state_extras = {
            "available_tools": runtime_inventory,
            "capability_tools": stable_tool_inventory(runtime_inventory, baseline_candidate),
            "trace_context_window_tokens": trace_context_window_tokens(),
            "trace_context_ratio": trace_context_ratio(),
            "evaluation_phase": example.get("evaluation_phase", "optimization"),
        }
        if critical_failures:
            state = {
                "messages": [],
                "candidate_runtime_skipped": True,
                "candidate_runtime_skip_reason": critical_failure_summary(critical_failures),
            }
        else:
            try:
                agent = create_deep_agent_from_application(runtime_application, llm)
                state = agent.invoke({"messages": messages_for_example(example)})
            except Exception as exc:
                state = {"messages": [], "error": exc}
        if not isinstance(state, dict):
            state = {"messages": getattr(state, "messages", [])}
        state.update(state_extras)
        state["candidate_hash"] = candidate_hash(candidate)
        state["candidate_excerpt"] = summarize_candidate(candidate)
        state["candidate_constraints"] = [constraint.__dict__ for constraint in constraints]
        state["candidate_metrics"] = candidate_metrics(candidate, baseline_candidate)
        state["baseline_response"] = (
            ""
            if critical_failures
            else run_configured_baseline_for_example(
                example,
                llm,
                project,
                baseline_candidate,
                mcp_loader,
            )
        )
        return state


def messages_for_example(example: dict[str, Any]) -> list[Any]:
    if example.get("messages"):
        messages = []
        for message in example["messages"]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages
    return [HumanMessage(content=example["input"])]


def run_configured_baseline_for_example(
    example: dict[str, Any],
    llm: BaseChatModel,
    project: DeepAgentProjectCandidate,
    baseline_candidate: dict[str, str],
    mcp_loader: Callable[[Sequence[MCPServerConfig], dict[str, str]], Sequence[BaseTool | Callable | dict[str, Any]]]
    | None = None,
) -> str:
    try:
        with tempfile.TemporaryDirectory(prefix="gepa_deep_agent_project_baseline_") as tmp:
            temp_root = Path(tmp)
            application = apply_candidate_to_deep_agent_project(project, baseline_candidate, temp_root)
            agent = create_deep_agent_from_configured_application(application, llm, mcp_loader=mcp_loader)
            state = agent.invoke({"messages": messages_for_example(example)})
            return last_message_text(state if isinstance(state, dict) else {"messages": []})
    except Exception as exc:
        return f"DRY_RUN_BASELINE_UNAVAILABLE: {type(exc).__name__}: {exc}"


def run_baseline_for_example(
    example: dict[str, Any],
    llm: BaseChatModel,
    seed_spec: DeepAgentTextSpec,
    surfaces: dict[str, ComponentSurface],
    baseline_candidate: dict[str, str],
) -> str:
    try:
        with tempfile.TemporaryDirectory(prefix="gepa_deep_agent_baseline_") as tmp:
            temp_root = Path(tmp)
            application = apply_candidate_to_deep_agent_spec(seed_spec, baseline_candidate, surfaces, temp_root)
            agent = create_deep_agent_from_application(application, llm)
            state = agent.invoke({"messages": [HumanMessage(content=example["input"])]})
            return last_message_text(state if isinstance(state, dict) else {"messages": []})
    except Exception as exc:
        return f"DRY_RUN_BASELINE_UNAVAILABLE: {type(exc).__name__}: {exc}"


def extract_route(text: str) -> str | None:
    match = ROUTE_RE.search(text)
    return match.group(1).strip().lower() if match else None


def trace_context_window_tokens() -> int:
    return _env_int("GEPA_CONTEXT_WINDOW_TOKENS", DEFAULT_CONTEXT_WINDOW_TOKENS)


def trace_context_ratio() -> float:
    return _env_float("GEPA_TRACE_CONTEXT_RATIO", DEFAULT_TRACE_CONTEXT_RATIO)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def trace_prompt_char_budget(state: dict[str, Any] | None = None) -> int:
    state = state or {}
    context_tokens = int(state.get("trace_context_window_tokens") or trace_context_window_tokens())
    ratio = float(state.get("trace_context_ratio") or trace_context_ratio())
    chars_per_token = _env_float("GEPA_TRACE_CHARS_PER_TOKEN", DEFAULT_TRACE_CHARS_PER_TOKEN)
    raw_budget = int(max(1, context_tokens) * max(0.01, min(0.80, ratio)) * max(0.5, chars_per_token))
    min_chars = _env_int("GEPA_TRACE_MIN_CHARS", DEFAULT_TRACE_MIN_CHARS)
    max_chars = _env_int("GEPA_TRACE_MAX_CHARS", DEFAULT_TRACE_MAX_CHARS)
    return max(min_chars, min(max_chars, raw_budget))


def message_content_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def trace_omit_tool_names() -> set[str]:
    configured = os.environ.get("GEPA_TRACE_OMIT_TOOL_NAMES")
    if configured is None:
        return set(DEFAULT_TRACE_OMIT_TOOL_NAMES)
    return {name.strip() for name in configured.split(",") if name.strip()}


def trace_tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, Mapping):
        return str(tool_call.get("name") or "unknown_tool")
    return str(getattr(tool_call, "name", None) or "unknown_tool")


def trace_tool_call_id(tool_call: Any) -> str | None:
    if isinstance(tool_call, Mapping):
        value = tool_call.get("id")
    else:
        value = getattr(tool_call, "id", None)
    return str(value) if value else None


def format_trace_tool_call(tool_call: Any) -> str:
    name = trace_tool_call_name(tool_call)
    if isinstance(tool_call, Mapping):
        args = tool_call.get("args")
    else:
        args = getattr(tool_call, "args", None)
    if args in (None, {}, ""):
        return name
    return f"{name} args={json.dumps(args, ensure_ascii=False, default=str)}"


def trace_lines(state: dict) -> list[str]:
    """Serialize the evaluation trace without mutating the rollout state.

    File-writing calls are implementation noise for skill/prompt evaluation, so
    their calls, arguments, and matching tool results are omitted by default.
    AI message text is always retained; an AI message with no text after tool
    filtering is represented explicitly instead of disappearing.
    """
    lines: list[str] = []
    omitted_tools = trace_omit_tool_names()
    messages = state.get("messages") or []
    omitted_tool_call_ids = {
        call_id
        for msg in messages
        for call in (getattr(msg, "tool_calls", None) or [])
        if trace_tool_call_name(call) in omitted_tools
        if (call_id := trace_tool_call_id(call)) is not None
    }
    for index, msg in enumerate(messages):
        msg_type = type(msg).__name__
        name = getattr(msg, "name", None)
        tool_call_id = getattr(msg, "tool_call_id", None)
        if msg_type == "ToolMessage" and (name in omitted_tools or tool_call_id in omitted_tool_call_ids):
            continue
        prefix = f"{index:04d} {msg_type}" + (f"[{name}]" if name else "")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None) or []
        visible_tool_calls = [call for call in tool_calls if trace_tool_call_name(call) not in omitted_tools]
        parts: list[str] = []
        if content:
            parts.append(message_content_text(content))
        if visible_tool_calls:
            parts.append("tool_calls=" + "; ".join(format_trace_tool_call(call) for call in visible_tool_calls))
        if msg_type == "AIMessage" and not parts:
            parts.append("[no textual content]")
        if not parts:
            continue
        lines.append(f"{prefix}: " + "\n".join(parts))
    if state.get("error") is not None:
        lines.append(f"ERROR: {type(state['error']).__name__}: {state['error']}")
    return lines


def full_trace_text(state: dict) -> str:
    return "\n".join(trace_lines(state))


def summarize_messages(
    state: dict,
    max_chars: int | None = None,
    summarizer: Callable[[str], str] | None = None,
) -> str:
    """Return a lossless trace or an LLM summary plus an untouched recent tail.

    Character slicing is intentionally not used. When no summarizer is
    available, the filtered trace is returned in full even if it exceeds the
    preferred prompt budget.
    """
    cached = state.get("evaluation_trace_summary")
    cached_budget = state.get("evaluation_trace_summary_budget")
    if isinstance(cached, str) and cached and (max_chars is None or cached_budget == max_chars):
        return cached
    max_chars = max_chars or trace_prompt_char_budget(state)
    lines = trace_lines(state)
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text
    if summarizer is None:
        return full_text
    return summarize_trace_with_model(lines, max_chars, summarizer)


def summarize_trace_with_model(
    lines: Sequence[str],
    max_chars: int,
    summarizer: Callable[[str], str],
) -> str:
    """Summarize older trace messages and preserve the recent tail verbatim."""
    keep_ratio = _env_float("GEPA_TRACE_KEEP_RATIO", DEFAULT_TRACE_KEEP_RATIO)
    keep_budget = int(max_chars * max(0.05, min(0.50, keep_ratio)))
    cutoff = trace_summary_cutoff(lines, keep_budget)
    older_lines = list(lines[:cutoff])
    recent_lines = list(lines[cutoff:])
    if not older_lines:
        older_lines = list(lines)
        recent_lines = []
    recent_text = "\n".join(recent_lines)
    summary_budget = max(1000, max_chars - len(recent_text) - 200)
    prompt = build_trace_summary_prompt("\n".join(older_lines), summary_budget)
    summary = str(summarizer(prompt)).strip()
    blocks = [f"<trace_summary>\n{summary}\n</trace_summary>"]
    if recent_text:
        blocks.append(f"<recent_trace>\n{recent_text}\n</recent_trace>")
    return "\n\n".join(blocks)


def trace_summary_cutoff(lines: Sequence[str], keep_budget: int) -> int:
    """Choose a whole-message cutoff while keeping a recent trace window."""
    used = 0
    cutoff = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        addition = len(lines[index]) + 1
        if used + addition > keep_budget:
            break
        used += addition
        cutoff = index
    if lines and cutoff == len(lines):
        return len(lines) - 1
    return cutoff


def build_trace_summary_prompt(trace: str, summary_budget: int) -> str:
    return (
        "Summarize the older portion of an AI-agent execution trace for a later evaluator and optimizer.\n"
        "Preserve concrete facts and chronology that affect whether the task was completed: AI conclusions and "
        "uncertainty, business-tool names, query intent and material arguments, material tool results, errors, missing "
        "evidence, and unfinished work. Distinguish observations from inference. Do not invent facts or optimization "
        "advice. Do not reproduce low-value file-writing mechanics. The recent trace will be appended unchanged.\n"
        f"Keep the summary concise, targeting no more than about {summary_budget} characters. Return summary text only.\n\n"
        f"<older_trace>\n{trace}\n</older_trace>"
    )


def prepare_evaluation_trace(
    state: dict[str, Any],
    summarizer: Callable[[str], str],
    max_chars: int | None = None,
) -> str:
    """Prepare and cache the inline trace passed to judge and reflection calls."""
    max_chars = max_chars or trace_prompt_char_budget(state)
    try:
        summary = summarize_messages(state, max_chars=max_chars, summarizer=summarizer)
        mode = "llm_summary" if summary.startswith("<trace_summary>") else "full"
    except Exception as exc:  # pragma: no cover - provider failures are runtime-specific.
        summary = full_trace_text(state)
        mode = f"summary_unavailable:{type(exc).__name__}"
    state["evaluation_trace_summary"] = summary
    state["evaluation_trace_summary_budget"] = max_chars
    state["evaluation_trace_mode"] = mode
    return summary


def summarize_candidate(candidate: dict[str, str]) -> dict[str, str]:
    return {name: text[:1200] for name, text in candidate.items()}


def candidate_metrics(candidate: dict[str, str], baseline_candidate: dict[str, str]) -> dict[str, Any]:
    def growth_for(name: str) -> float:
        baseline_len = len(baseline_candidate.get(name, ""))
        return (len(candidate[name]) - baseline_len) / max(1, baseline_len)

    return {
        "lengths": {name: len(text) for name, text in candidate.items()},
        "growth": {name: growth_for(name) for name in candidate},
    }


def candidate_hash(candidate: dict[str, str]) -> str:
    payload = json.dumps(sorted(candidate.items()), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_candidate_constraints(
    candidate: dict[str, str],
    baseline_candidate: dict[str, str],
    surfaces: dict[str, ComponentSurface],
    materialized_root: Path | None = None,
) -> list[ConstraintResult]:
    results: list[ConstraintResult] = []
    public_texts: list[str] = []
    for key, text in candidate.items():
        surface = surfaces[key]
        limit = MAX_COMPONENT_CHARS.get(surface.source_type, 12000)
        growth = (len(text) - len(baseline_candidate.get(key, ""))) / max(1, len(baseline_candidate.get(key, "")))
        results.extend(
            [
                ConstraintResult(bool(text.strip()), f"{key}:non_empty", "component is non-empty"),
                ConstraintResult(len(text) <= limit, f"{key}:size_limit", f"{len(text)}/{limit} chars"),
                ConstraintResult(
                    growth <= MAX_PROMPT_GROWTH,
                    f"{key}:growth_limit",
                    f"{growth:+.1%} growth, max {MAX_PROMPT_GROWTH:+.1%}",
                    "advisory",
                ),
            ]
        )
        if surface.source_type in {"skill", "reference", "memory", "prompt"}:
            public_texts.append(text)
        if surface.source_type == "description":
            results.append(
                ConstraintResult(
                    len(text.split()) >= 4,
                    f"{key}:description_detail",
                    "description has detail",
                    "advisory",
                )
            )
        if surface.source_type == "skill":
            results.extend(skill_structure_constraints(key, text))
            if materialized_root is not None:
                results.extend(skill_script_reference_constraints(key, text, surface, materialized_root))
        results.extend(component_boundary_constraints(key, text, candidate, surfaces))

    runtime_matches = [m.group(0) for m in RUNTIME_SPECIFIC_PATTERN.finditer("\n".join(public_texts))]
    results.append(
        ConstraintResult(
            not runtime_matches,
            "runtime_neutrality",
            "runtime-neutral" if not runtime_matches else f"runtime-specific terms: {runtime_matches[:5]}",
        )
    )
    return results


def component_boundary_constraints(
    key: str,
    text: str,
    candidate: dict[str, str],
    surfaces: dict[str, ComponentSurface],
) -> list[ConstraintResult]:
    """High-confidence component-boundary checks.

    Keep this intentionally conservative. Ambiguous style and scope questions
    belong to the reflection judge, not deterministic gates.
    """
    surface = surfaces[key]
    stripped = text.strip()
    results: list[ConstraintResult] = []

    if surface.source_type in {"prompt", "description"}:
        has_skill_frontmatter = bool(YAML_FRONTMATTER_RE.search(stripped))
        results.append(
            ConstraintResult(
                not has_skill_frontmatter,
                f"{key}:boundary:no_skill_frontmatter",
                "component does not embed SKILL.md YAML frontmatter"
                if not has_skill_frontmatter
                else "component embeds SKILL.md-style YAML frontmatter",
            )
        )

    if surface.source_type != "reference":
        has_component_labels = contains_candidate_component_label(key, text, candidate)
        results.append(
            ConstraintResult(
                not has_component_labels,
                f"{key}:boundary:no_component_labels",
                "component does not include candidate excerpt labels"
                if not has_component_labels
                else "component includes candidate excerpt labels",
            )
        )
    return results


def contains_candidate_component_label(key: str, text: str, candidate: dict[str, str]) -> bool:
    del key
    for component_key in candidate:
        if re.search(rf"(?m)^\s*(?:#{{1,6}}\s*)?{re.escape(component_key)}\s*$", text):
            return True
    return bool(COMPONENT_LABEL_RE.search(text))


def skill_script_reference_constraints(
    key: str,
    text: str,
    surface: ComponentSurface,
    materialized_root: Path,
) -> list[ConstraintResult]:
    if surface.relative_path is None:
        return []
    skill_dir = (materialized_root / surface.relative_path).parent
    results: list[ConstraintResult] = []
    for script_ref in referenced_script_paths(text):
        source = skill_dir / script_ref
        alias = materialized_root / script_ref
        exists = source.exists()
        alias_matches = alias.exists() and source.exists() and alias.read_bytes() == source.read_bytes()
        results.append(
            ConstraintResult(
                exists and alias_matches,
                f"{key}:script:{script_ref.as_posix()}",
                "referenced script is materialized and executable from workspace root"
                if exists and alias_matches
                else f"referenced script or root alias missing for {skill_dir.relative_to(materialized_root).as_posix()}",
            )
        )
    return results


def skill_structure_constraints(key: str, text: str) -> list[ConstraintResult]:
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    frontmatter_data: dict[str, Any] | None = None
    if frontmatter_match is not None:
        try:
            parsed = yaml.safe_load(frontmatter_match.group(1))
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, dict):
            frontmatter_data = parsed
    skill_name = str((frontmatter_data or {}).get("name", "")).strip()
    description = str((frontmatter_data or {}).get("description", "")).strip()
    expected_name = key.rsplit(":", maxsplit=2)[-2]
    valid_name = (
        bool(skill_name)
        and len(skill_name) <= 64
        and skill_name == expected_name
        and not skill_name.startswith("-")
        and not skill_name.endswith("-")
        and "--" not in skill_name
        and all(
            character == "-" or character.isdigit() or (character.isalpha() and character.islower())
            for character in skill_name
        )
    )
    lowered = text.lower()
    has_ordered_workflow = (
        "workflow" in lowered or "工作流程" in text or re.search(r"^\s*1[.、]", text, re.M) is not None
    )
    has_failure_modes = any(marker in lowered for marker in ["failure", "fallback", "if "]) or any(
        marker in text for marker in ["失败模式", "异常处理", "如果", "若", "当"]
    )
    has_guardrails = any(marker in lowered for marker in ["do not", "never", "avoid", "guardrail", "blacklist"]) or any(
        marker in text for marker in ["不得", "禁止", "避免", "约束", "护栏", "不应"]
    )
    return [
        ConstraintResult(
            frontmatter_match is not None,
            f"{key}:frontmatter",
            "SKILL.md starts with a complete YAML frontmatter block",
        ),
        ConstraintResult(
            frontmatter_data is not None,
            f"{key}:frontmatter_yaml",
            "SKILL.md frontmatter is a valid YAML mapping",
        ),
        ConstraintResult(
            bool(skill_name and description),
            f"{key}:name_description",
            "frontmatter includes non-empty name and description",
        ),
        ConstraintResult(
            valid_name,
            f"{key}:name_validation",
            f"frontmatter name matches skill directory '{expected_name}' and follows Agent Skills naming rules",
            "advisory",
        ),
        ConstraintResult(
            has_ordered_workflow,
            f"{key}:workflow",
            "skill includes ordered workflow",
            "advisory",
        ),
        ConstraintResult(
            has_failure_modes,
            f"{key}:failure_modes",
            "skill includes failure modes or if-then branches",
            "advisory",
        ),
        ConstraintResult(
            has_guardrails,
            f"{key}:risk_blacklist",
            "skill includes do-not/guardrail guidance",
            "advisory",
        ),
    ]


def hard_constraint_failures(state: dict) -> list[dict[str, Any]]:
    return [
        c
        for c in state.get("candidate_constraints", [])
        if not c.get("passed") and str(c.get("severity", "hard")) == "hard"
    ]


def critical_constraint_results(constraints: Sequence[ConstraintResult]) -> list[ConstraintResult]:
    return [
        constraint
        for constraint in constraints
        if not constraint.passed
        and constraint.severity == "hard"
        and is_critical_constraint_failure(constraint.__dict__)
    ]


def critical_failure_summary(failures: Sequence[ConstraintResult]) -> str:
    return "; ".join(f"{failure.name}: {failure.message}" for failure in failures)


def is_critical_constraint_failure(failure: dict[str, Any]) -> bool:
    name = str(failure.get("name", ""))
    return (
        name.endswith(":non_empty")
        or name.endswith(":size_limit")
        or name.endswith(":growth_limit")
        or name.endswith(":frontmatter")
        or name.endswith(":frontmatter_yaml")
        or name.endswith(":name_description")
        or "runtime_neutrality" in name
        or ":script:" in name
        or ":boundary:" in name
    )


def constraint_gate_penalty(failures: list[dict[str, Any]]) -> float:
    if not failures:
        return 0.0
    penalty = min(0.45, 0.04 * len(failures))
    if any(is_critical_constraint_failure(failure) for failure in failures):
        penalty = max(penalty, 0.35)
    if any("runtime_neutrality" in str(failure.get("name", "")) for failure in failures):
        penalty += 0.25
    return min(0.80, penalty)


def correctness_cap(response: str, expected: str) -> float:
    if not expected:
        return 1.0 if response.strip() else 0.35
    predicted = extract_route(response)
    if predicted == expected:
        return 1.0
    if predicted is None:
        return 0.40
    return 0.55


def rubric_checkpoints(example: dict[str, Any]) -> list[Any]:
    metadata = example.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    raw_checkpoints = metadata.get("checkpoints") or metadata.get("rubric_checkpoints") or []
    return list(raw_checkpoints) if isinstance(raw_checkpoints, list) else []


def normalize_for_keyword_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE)).strip()


def example_data_text(example: dict[str, Any], limit: int = 4000) -> str:
    data = str(example.get("data") or "").strip()
    if not data:
        return "n/a"
    if len(data) <= limit:
        return data
    return data[:limit] + "\n...[truncated]"


def checkpoint_label(checkpoint: Any) -> str:
    if isinstance(checkpoint, Mapping):
        return str(checkpoint.get("label") or checkpoint.get("name") or checkpoint.get("id") or "checkpoint")
    return str(checkpoint)


def checkpoint_keywords(checkpoint: Any) -> list[str]:
    if isinstance(checkpoint, Mapping):
        keywords = checkpoint.get("keywords") or checkpoint.get("aliases") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        values = [str(item) for item in keywords if str(item).strip()]
        label = checkpoint_label(checkpoint)
        return values or [label]
    return [str(checkpoint)]


def checkpoint_matches(response_text: str, checkpoint: Any) -> bool:
    normalized_response = normalize_for_keyword_match(response_text)
    for keyword in checkpoint_keywords(checkpoint):
        normalized_keyword = normalize_for_keyword_match(keyword)
        if normalized_keyword and normalized_keyword in normalized_response:
            return True
    return False


def rubric_checkpoint_results(example: dict[str, Any], response: str) -> tuple[list[str], list[str], float]:
    checkpoints = rubric_checkpoints(example)
    if not checkpoints:
        return [], [], 1.0
    matched: list[str] = []
    missing: list[str] = []
    for checkpoint in checkpoints:
        label = checkpoint_label(checkpoint)
        if checkpoint_matches(response, checkpoint):
            matched.append(label)
        else:
            missing.append(label)
    return matched, missing, len(matched) / max(1, len(checkpoints))


def trace_expectations(example: dict[str, Any]) -> list[Any]:
    metadata = example.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    raw_expectations = metadata.get("trace_expectations") or metadata.get("data_acquisition_expectations") or []
    return list(raw_expectations) if isinstance(raw_expectations, list) else []


def trace_expectation_label(expectation: Any) -> str:
    if isinstance(expectation, Mapping):
        return str(expectation.get("label") or expectation.get("name") or expectation.get("id") or "trace_expectation")
    return str(expectation)


def trace_expectation_keywords(expectation: Any) -> list[str]:
    if isinstance(expectation, Mapping):
        keywords = (
            expectation.get("tool_intent_keywords") or expectation.get("keywords") or expectation.get("aliases") or []
        )
        if isinstance(keywords, str):
            keywords = [keywords]
        values = [str(item) for item in keywords if str(item).strip()]
        label = trace_expectation_label(expectation)
        return values or [label]
    return [str(expectation)]


def trace_expectation_tool_names(expectation: Any) -> list[str]:
    if not isinstance(expectation, Mapping):
        return []
    raw_names = expectation.get("tool_names") or expectation.get("tools") or expectation.get("required_tools") or []
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    return [str(name).strip() for name in raw_names if str(name).strip()]


def tool_result_is_successful(message: Any) -> tuple[bool, str]:
    status = getattr(message, "status", None)
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    status = status or additional_kwargs.get("status")
    normalized_status = str(status or "").strip().lower()
    content = message_content_text(getattr(message, "content", "")).strip()
    if normalized_status in {"error", "failed", "failure"}:
        return False, normalized_status
    if not content:
        return False, normalized_status or "empty_result"
    if TOOL_FAILURE_PATTERN.search(content):
        return False, normalized_status or "error_result"
    return True, normalized_status or "success"


def trace_tool_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return paired tool calls/results; AI prose never counts as acquisition."""
    calls_by_id: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    omitted_tools = trace_omit_tool_names()
    for message in state.get("messages") or []:
        for call in getattr(message, "tool_calls", None) or []:
            call_id = trace_tool_call_id(call)
            if call_id is None:
                continue
            if isinstance(call, Mapping):
                args = call.get("args")
            else:
                args = getattr(call, "args", None)
            calls_by_id[call_id] = {
                "tool_call_id": call_id,
                "name": trace_tool_call_name(call),
                "args": args,
            }
        if not isinstance(message, ToolMessage) and type(message).__name__ != "ToolMessage":
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        call = calls_by_id.get(tool_call_id, {})
        name = str(getattr(message, "name", None) or call.get("name") or "unknown_tool")
        if name in omitted_tools:
            continue
        success, status = tool_result_is_successful(message)
        evidence.append(
            {
                "tool_call_id": tool_call_id or None,
                "name": name,
                "args": call.get("args"),
                "result": message_content_text(getattr(message, "content", "")),
                "success": success,
                "status": status,
            }
        )
    return evidence


def tool_name_matches(actual: str, expected: str) -> bool:
    actual_normalized = actual.strip().lower()
    expected_normalized = expected.strip().lower()
    return actual_normalized == expected_normalized or actual_normalized.endswith(
        (f"::{expected_normalized}", f".{expected_normalized}", f"/{expected_normalized}")
    )


def inventory_for_tool(
    tool_name: str,
    inventory: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    return [item for item in inventory if tool_name_matches(str(item.get("name", "")), tool_name)]


def tool_evidence_matches_expectation(
    expectation: Any,
    evidence: Mapping[str, Any],
    inventory: Sequence[Mapping[str, str]],
) -> bool:
    if not evidence.get("success"):
        return False
    tool_name = str(evidence.get("name") or "")
    explicit_names = trace_expectation_tool_names(expectation)
    if explicit_names:
        return any(tool_name_matches(tool_name, expected_name) for expected_name in explicit_names)

    matching_inventory = inventory_for_tool(tool_name, inventory)
    if not expectation_supported_by_tools(expectation, matching_inventory):
        return False
    evidence_text = normalize_for_keyword_match(
        " ".join(
            [
                tool_name,
                json.dumps(evidence.get("args"), ensure_ascii=False, default=str),
                str(evidence.get("result") or ""),
            ]
        )
    )
    return any(
        normalized_keyword and normalized_keyword in evidence_text
        for keyword in trace_expectation_keywords(expectation)
        if (normalized_keyword := normalize_for_keyword_match(keyword))
    )


def trace_expectation_matches(
    example: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, list[str]], list[dict[str, Any]]]:
    expectations = trace_expectations(example)
    inventory = list(state.get("capability_tools") or state.get("available_tools") or [])
    tool_evidence = trace_tool_evidence(state)
    matched: list[str] = []
    missing: list[str] = []
    evidence_by_expectation: dict[str, list[str]] = {}
    for expectation in expectations:
        label = trace_expectation_label(expectation)
        matching_tools = [
            str(item.get("name") or "unknown_tool")
            for item in tool_evidence
            if tool_evidence_matches_expectation(expectation, item, inventory)
        ]
        if matching_tools:
            matched.append(label)
            evidence_by_expectation[label] = sorted(set(matching_tools))
        else:
            missing.append(label)
    return matched, missing, evidence_by_expectation, tool_evidence


def trace_expectation_results(example: dict[str, Any], state: dict[str, Any]) -> tuple[list[str], list[str], float]:
    expectations = trace_expectations(example)
    if not expectations:
        return [], [], 1.0
    matched, missing, _evidence_by_expectation, _tool_evidence = trace_expectation_matches(example, state)
    return matched, missing, len(matched) / max(1, len(expectations))


def expectation_supported_by_tools(expectation: Any, inventory: Sequence[Mapping[str, str]]) -> bool:
    if not inventory:
        return False
    explicit_names = trace_expectation_tool_names(expectation)
    if explicit_names:
        return any(
            tool_name_matches(str(item.get("name", "")), expected_name)
            for item in inventory
            for expected_name in explicit_names
        )
    inventory_text = normalize_for_keyword_match(tool_inventory_text(inventory))
    if not inventory_text:
        return False
    normalized_keywords = {
        normalized_keyword
        for keyword in trace_expectation_keywords(expectation)
        if (normalized_keyword := normalize_for_keyword_match(keyword))
    }
    matched_keywords = {keyword for keyword in normalized_keywords if keyword in inventory_text}
    required_matches = min(2, len(normalized_keywords))
    return bool(required_matches) and len(matched_keywords) >= required_matches


def data_acquisition_diagnostics(example: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    expectations = trace_expectations(example)
    if not expectations:
        return {
            "matched_trace_expectations": [],
            "missing_trace_expectations": [],
            "trace_expectation_coverage": 1.0,
            "tool_supported_missing_expectations": [],
            "tool_capability_gaps": [],
            "trace_expectation_evidence": {},
            "successful_tool_evidence": [],
            "failed_tool_evidence": [],
        }
    matched, missing, evidence_by_expectation, tool_evidence = trace_expectation_matches(example, state)
    coverage = len(matched) / max(1, len(expectations))
    missing_set = set(missing)
    inventory = list(state.get("capability_tools") or state.get("available_tools") or [])
    tool_supported: list[str] = []
    capability_gaps: list[str] = []
    for expectation in expectations:
        label = trace_expectation_label(expectation)
        if label not in missing_set:
            continue
        if expectation_supported_by_tools(expectation, inventory):
            tool_supported.append(label)
        else:
            capability_gaps.append(label)
    return {
        "matched_trace_expectations": matched,
        "missing_trace_expectations": missing,
        "trace_expectation_coverage": coverage,
        "tool_supported_missing_expectations": tool_supported,
        "tool_capability_gaps": capability_gaps,
        "trace_expectation_evidence": evidence_by_expectation,
        "successful_tool_evidence": [
            {
                **item,
                "result": str(item.get("result") or "")[:1200],
            }
            for item in tool_evidence
            if item.get("success")
        ],
        "failed_tool_evidence": [
            {
                **item,
                "result": str(item.get("result") or "")[:1200],
            }
            for item in tool_evidence
            if not item.get("success")
        ],
    }


def tool_evidence_text(evidence: Sequence[Mapping[str, Any]], limit: int = 12) -> str:
    lines = []
    for item in evidence[:limit]:
        args = json.dumps(item.get("args"), ensure_ascii=False, default=str)
        result = str(item.get("result") or "").replace("\n", " ")[:300]
        lines.append(
            f"- {item.get('name', 'unknown_tool')} status={item.get('status', 'unknown')} args={args} result={result}"
        )
    return "\n".join(lines) or "- none"


def rubric_coverage_cap(example: dict[str, Any], response: str) -> float:
    checkpoints = rubric_checkpoints(example)
    if example.get("answer") or example.get("expected") or not checkpoints:
        return 1.0
    if not response.strip():
        return 0.25
    _matched, _missing, coverage = rubric_checkpoint_results(example, response)
    if coverage >= 1.0:
        return 1.0
    if coverage >= 0.80:
        return 0.90
    if coverage >= 0.60:
        return 0.75
    if coverage >= 0.40:
        return 0.60
    return 0.45


def constraint_cap(failures: list[dict[str, Any]]) -> float:
    if any(is_critical_constraint_failure(failure) for failure in failures):
        return 0.0
    if failures:
        return 0.80
    return 1.0


def structure_score(state: dict) -> float:
    constraints = [
        constraint
        for constraint in state.get("candidate_constraints", [])
        if str(constraint.get("severity", "hard")) == "hard"
    ]
    if not constraints:
        return 0.0
    return sum(1 for c in constraints if c.get("passed")) / len(constraints)


def output_score(response: str, expected: str) -> float:
    if not expected:
        return 1.0 if response.strip() else 0.0
    predicted = extract_route(response)
    if predicted == expected:
        return 1.0
    if predicted is not None:
        return 0.35
    return 0.0


def hard_score(response: str, expected: str) -> float:
    if not expected:
        return 1.0 if response.strip() else 0.0
    return 1.0 if extract_route(response) == expected else 0.0


def mixed_gate_score(hard: float, soft: float, mixed_weight: float = 0.5) -> float:
    weight = max(0.0, min(1.0, mixed_weight))
    return (1.0 - weight) * hard + weight * soft


def effect_score(response: str, baseline_response: str, expected: str) -> tuple[float, str]:
    if baseline_response.startswith("DRY_RUN_BASELINE_UNAVAILABLE"):
        return 0.35 + 0.4 * output_score(response, expected), "dry_run"
    candidate_score = output_score(response, expected)
    baseline_score = output_score(baseline_response, expected)
    if candidate_score > baseline_score:
        return 1.0, "full_test"
    if candidate_score == baseline_score and candidate_score == 1.0:
        return 0.85, "full_test"
    if candidate_score == baseline_score:
        return 0.45, "full_test"
    return 0.0, "full_test"


def specificity_score(candidate: dict[str, str]) -> float:
    total = len(candidate) or 1
    soft = sum(1 for text in candidate.values() if SOFTENER_PATTERN.search(text))
    return max(0.0, 1.0 - soft / total)


def evaluate_response(example: dict[str, Any], state: dict) -> tuple[float, str]:
    if state.get("error") is not None:
        feedback = f"Rollout failed: {type(state['error']).__name__}: {state['error']}"
        return 0.0, feedback

    response = last_message_text(state)
    expected = example.get("answer") or example.get("expected") or ""
    baseline_response = state.get("baseline_response", "")
    hard = hard_score(response, expected)
    soft = output_score(response, expected)
    mixed = mixed_gate_score(hard, soft)
    baseline_hard = hard_score(baseline_response, expected)
    baseline_soft = output_score(baseline_response, expected)
    baseline_mixed = mixed_gate_score(baseline_hard, baseline_soft)
    struct = structure_score(state)
    effect, eval_mode = effect_score(response, baseline_response, expected)
    specificity = specificity_score(state.get("candidate_excerpt", {}))
    failures = hard_constraint_failures(state)
    gate_penalty = constraint_gate_penalty(failures)
    raw_composite = max(0.0, 0.45 * effect + 0.35 * struct + 0.20 * specificity - gate_penalty)
    answer_cap = correctness_cap(response, expected)
    rubric_cap = rubric_coverage_cap(example, response)
    matched_checkpoints, missing_checkpoints, rubric_coverage = rubric_checkpoint_results(example, response)
    acquisition_diagnostics = data_acquisition_diagnostics(example, state)
    gate_cap = constraint_cap(failures)
    composite = min(raw_composite, answer_cap, rubric_cap, gate_cap)
    fitness = {
        "hard": hard,
        "soft": soft,
        "mixed": mixed,
        "baseline_hard": baseline_hard,
        "baseline_soft": baseline_soft,
        "baseline_mixed": baseline_mixed,
        "effect": effect,
        "structure": struct,
        "specificity": specificity,
        "gate_penalty": gate_penalty,
        "correctness_cap": answer_cap,
        "rubric_cap": rubric_cap,
        "rubric_coverage": rubric_coverage,
        "matched_rubric_checkpoints": matched_checkpoints,
        "missing_rubric_checkpoints": missing_checkpoints,
        **acquisition_diagnostics,
        "constraint_cap": gate_cap,
        "raw_composite": raw_composite,
        "eval_mode": eval_mode,
        "composite": composite,
    }
    fitness.update(mutation_eligibility_diagnostics(example, state, failures, fitness))
    failure_classification, classification_reason = classify_failure(example, state, response, failures, fitness)
    fitness["failure_classification"] = failure_classification
    fitness["classification_reason"] = classification_reason
    state["fitness"] = fitness
    feedback = build_feedback(example, state, response, baseline_response, failures, fitness)
    return composite, feedback


def evaluate_response_with_judge(
    example: dict[str, Any],
    state: dict,
    judge_lm: Callable[[str], str],
) -> tuple[float, str]:
    """Use the reflection model as the main evaluator, with hard rules as caps."""
    prepare_evaluation_trace(state, judge_lm)
    deterministic_score, deterministic_feedback = evaluate_response(example, state)
    if state.get("error") is not None:
        return deterministic_score, deterministic_feedback

    failures = hard_constraint_failures(state)
    if constraint_cap(failures) == 0.0:
        return (
            deterministic_score,
            f"{deterministic_feedback}\n\nReflection judge skipped: a deterministic critical constraint rejected "
            "the candidate before runtime.",
        )
    prompt = build_judge_prompt(example, state, deterministic_score, deterministic_feedback, failures)
    try:
        raw_judge = judge_lm(prompt)
    except Exception as exc:  # pragma: no cover - defensive fallback for flaky judge providers.
        return (
            deterministic_score,
            f"{deterministic_feedback}\n\nReflection judge unavailable: {type(exc).__name__}: {exc}",
        )

    payload = parse_judge_json(raw_judge)
    if payload is None:
        return (
            deterministic_score,
            f"{deterministic_feedback}\n\nReflection judge returned non-JSON output:\n{raw_judge[:1200]}",
        )

    candidate = state.get("candidate_excerpt", {})
    response = last_message_text(state)
    expected = example.get("answer") or example.get("expected") or ""
    failure_classification, default_reason = classify_failure(
        example, state, response, failures, state.get("fitness", {})
    )
    suggested = str(payload.get("suggested_component") or "").strip()
    mutation_eligible = bool(state.get("fitness", {}).get("mutation_eligible", True))
    if not mutation_eligible:
        suggested = ""
    elif suggested not in candidate:
        suggested = suggest_component_to_update(
            state,
            weakest_dimension(state.get("fitness", {}), failures),
            failure_classification,
            expected,
        )

    raw_score = coerce_score(payload.get("score"), deterministic_score)
    correctness = correctness_cap(response, expected)
    rubric_cap_value = rubric_coverage_cap(example, response)
    gate_cap = constraint_cap(failures)
    cap = min(correctness, rubric_cap_value, gate_cap)
    score = min(raw_score, cap)
    fitness = dict(state.get("fitness", {}))
    fitness.update(
        {
            "judge_score": raw_score,
            "judge_correctness_cap": correctness,
            "judge_rubric_cap": rubric_cap_value,
            "judge_constraint_cap": gate_cap,
            "judge_cap": cap,
            "eval_mode": "llm_judge",
            "composite": score,
        }
    )
    state["fitness"] = fitness

    judged_classification = str(payload.get("failure_classification") or failure_classification).strip()
    if judged_classification not in {
        SKILL_DEFECT,
        EXECUTION_LAPSE,
        TOOL_CAPABILITY_GAP,
        INSUFFICIENT_RUNTIME_EVIDENCE,
        NO_FAILURE,
    }:
        judged_classification = failure_classification
    classification_reason = str(payload.get("classification_reason") or default_reason)
    tool_gaps = [str(item) for item in fitness.get("tool_capability_gaps") or []]
    supported_missing = [str(item) for item in fitness.get("tool_supported_missing_expectations") or []]
    _matched, missing_checkpoints, _coverage = rubric_checkpoint_results(example, response)
    if failures:
        judged_classification = SKILL_DEFECT
    elif missing_checkpoints and not expected:
        if supported_missing:
            judged_classification = EXECUTION_LAPSE
            classification_reason = "agent skipped available data-acquisition paths for: " + ", ".join(
                supported_missing[:3]
            )
        elif not mutation_eligible:
            judged_classification = str(fitness.get("mutation_block_classification") or INSUFFICIENT_RUNTIME_EVIDENCE)
            classification_reason = str(
                fitness.get("mutation_eligibility_reason")
                or "missing expert checkpoints are not supported by runtime-observable evidence"
            )
            suggested = ""
        elif judged_classification in {
            TOOL_CAPABILITY_GAP,
            INSUFFICIENT_RUNTIME_EVIDENCE,
            NO_FAILURE,
        }:
            judged_classification = SKILL_DEFECT
            classification_reason = "runtime evidence exists, but reusable analysis missed: " + ", ".join(
                missing_checkpoints[:3]
            )
    elif supported_missing and not expected:
        judged_classification = EXECUTION_LAPSE
        classification_reason = "agent skipped available data-acquisition paths for: " + ", ".join(
            supported_missing[:3]
        )
    elif tool_gaps:
        judged_classification = TOOL_CAPABILITY_GAP
        suggested = ""
        classification_reason = "required evidence is unavailable from current tools: " + ", ".join(tool_gaps[:3])
    elif judged_classification == NO_FAILURE:
        suggested = ""
    if (
        mutation_eligible
        and judged_classification not in {TOOL_CAPABILITY_GAP, INSUFFICIENT_RUNTIME_EVIDENCE, NO_FAILURE}
        and suggested not in candidate
    ):
        suggested = suggest_component_to_update(
            state,
            weakest_dimension(fitness, failures),
            judged_classification,
            expected,
        )
    if not mutation_eligible:
        suggested = ""
    fitness["failure_classification"] = judged_classification
    fitness["classification_reason"] = classification_reason
    state["fitness"] = fitness
    return score, build_judge_feedback(
        example=example,
        state=state,
        response=response,
        baseline_response=state.get("baseline_response", ""),
        failures=failures,
        deterministic_feedback=deterministic_feedback,
        raw_judge=raw_judge,
        payload=payload,
        score=score,
        raw_score=raw_score,
        cap=cap,
        correctness_cap_value=correctness,
        rubric_cap_value=rubric_cap_value,
        constraint_cap_value=gate_cap,
        failure_classification=judged_classification,
        classification_reason=classification_reason,
        suggested=suggested,
    )


def build_judge_prompt(
    example: dict[str, Any],
    state: dict,
    deterministic_score: float,
    deterministic_feedback: str,
    failures: list[dict[str, Any]],
) -> str:
    candidate = state.get("candidate_excerpt", {})
    hard_lines = "\n".join(f"- {f['name']}: {f['message']}" for f in failures) or "- none"
    advisory = [
        c
        for c in state.get("candidate_constraints", [])
        if not c.get("passed") and str(c.get("severity", "hard")) == "advisory"
    ]
    advisory_lines = "\n".join(f"- {c['name']}: {c['message']}" for c in advisory[:20]) or "- none"
    matched_checkpoints, missing_checkpoints, rubric_coverage = rubric_checkpoint_results(
        example,
        last_message_text(state),
    )
    checkpoint_lines = "\n".join(f"- {checkpoint_label(item)}" for item in rubric_checkpoints(example)) or "- none"
    missing_checkpoint_lines = "\n".join(f"- {item}" for item in missing_checkpoints) or "- none"
    acquisition_diagnostics = data_acquisition_diagnostics(example, state)
    matched_trace_expectations = acquisition_diagnostics["matched_trace_expectations"]
    missing_trace_expectations = acquisition_diagnostics["missing_trace_expectations"]
    trace_coverage = acquisition_diagnostics["trace_expectation_coverage"]
    trace_expectation_lines = (
        "\n".join(f"- {trace_expectation_label(item)}" for item in trace_expectations(example)) or "- none"
    )
    missing_trace_expectation_lines = "\n".join(f"- {item}" for item in missing_trace_expectations) or "- none"
    matched_trace_expectation_lines = "\n".join(f"- {item}" for item in matched_trace_expectations) or "- none"
    tool_supported_missing_lines = (
        "\n".join(f"- {item}" for item in acquisition_diagnostics["tool_supported_missing_expectations"]) or "- none"
    )
    tool_capability_gap_lines = (
        "\n".join(f"- {item}" for item in acquisition_diagnostics["tool_capability_gaps"]) or "- none"
    )
    successful_tool_evidence_lines = tool_evidence_text(acquisition_diagnostics["successful_tool_evidence"])
    failed_tool_evidence_lines = tool_evidence_text(acquisition_diagnostics["failed_tool_evidence"])
    available_tools = tool_inventory_text(state.get("available_tools") or []) or "- none"
    fitness = state.get("fitness", {})
    mutation_eligible = bool(fitness.get("mutation_eligible", True))
    mutation_eligibility_reason = str(fitness.get("mutation_eligibility_reason") or "not evaluated")
    return (
        "You are the evaluator for a Deep Agents GEPA text-surface optimization run.\n"
        "Use hard constraints as non-negotiable validity rules. Treat advisory notes as hints, not automatic failures.\n"
        "Score whether the candidate behavior and text surfaces improved for the task, then recommend the single best "
        "component to edit next.\n"
        "Choose the component by ownership: AGENTS.md/system prompts hold stable global execution policy; SKILL.md holds "
        "invariant workflow, resource routing, failure modes, and guardrails; reference/*.md holds scoped domain "
        "methodology, industry patterns, calculations, and expert knowledge; tool descriptions hold invocation semantics "
        "and capability boundaries. Prefer the most specific existing reference component for domain knowledge. Do not "
        "grow SKILL.md into an industry catalog.\n"
        "Treat a single example as evidence for a candidate rule, not proof of a universal rule. Recommend explicit "
        "applicability signals and exclusions whenever a change could help one industry or business model but harm "
        "another. A useful rule must be operational: trigger, evidence, analysis/comparison, risk transmission, and "
        "approval or verification action. A trigger is a conditional observable signal, not a universal checklist; "
        "evidence is a borrower-specific acquisition plan whose source and comparison baseline may vary by business "
        "model. Keep unsupported or uncollected evidence as a hypothesis. Reject vague advice that lacks these elements.\n"
        "Data-acquisition expectations are satisfied only by paired, successful tool-call results whose declared seed "
        "capability matches the expectation. Keywords in prompts, skills, AI prose, or final answers are not acquisition "
        "evidence. A hidden expert checkpoint is allowed to lower the task score, but it may drive a text mutation only "
        "when the trace contains relevant successful evidence or the agent skipped an available matching tool. If "
        "current tools cannot obtain the evidence, classify TOOL_CAPABILITY_GAP. If the evaluator cannot establish "
        "whether the lesson is observable at runtime, classify INSUFFICIENT_RUNTIME_EVIDENCE. In both cases leave "
        "suggested_component empty. Never convert unavailable company facts into memorized skill/reference content.\n"
        "For NO_FAILURE, leave suggested_component empty. Successful examples are regression constraints and positive "
        "evidence; they must not vote to mutate a component.\n"
        "If Expected is not `rubric-only`, treat it as the authoritative target label, route, answer, or structured "
        "result. Do not reinterpret the task as solving the user's underlying real-world problem unless the rubric "
        "explicitly asks for that. For routing or classification tasks, score the final response by whether it returns "
        "the expected label/route and recommend text-surface changes that improve that classification behavior. "
        "Operational troubleshooting advice instead of the expected label is a failure.\n"
        "For rubric-only expert-experience examples, use rubric checkpoints as a strict behavior-scoring checklist: "
        "missing checkpoints still cap the score. Keep scoring separate from mutation eligibility. Expert evaluation "
        "data is evaluator-only material; do not copy its case facts, exact conclusions, thresholds, or complete "
        "analysis into a runtime component. When runtime evidence supports a reusable lesson, recommend only the "
        "smallest non-obvious cue. For learned references, prefer a compact condition -> concern -> consequence reminder "
        "and rely on the runtime model's general knowledge for standard investigation and analysis. Never tell the "
        "runtime agent to read Expert data, rubrics, checkpoints, or evaluator feedback; those fields are hidden during "
        "rollout.\n\n"
        "Return JSON only, with this schema:\n"
        "{\n"
        '  "score": 0.0,\n'
        f'  "failure_classification": "{SKILL_DEFECT}|{EXECUTION_LAPSE}|{TOOL_CAPABILITY_GAP}|'
        f'{INSUFFICIENT_RUNTIME_EVIDENCE}|{NO_FAILURE}",\n'
        '  "classification_reason": "short reason",\n'
        '  "mutation_eligible": false,\n'
        '  "suggested_component": "one key from allowed_components, or empty when mutation_eligible is false",\n'
        '  "suggested_component_reason": "short reason",\n'
        '  "reusable_lesson": "smallest non-obvious reusable lesson, or empty",\n'
        '  "knowledge_scope": "global_policy|invariant_workflow|scoped_domain_rule|tool_semantics",\n'
        '  "applicability_scope": "observable triggers, relevant scopes, and exclusions",\n'
        '  "cross_case_regression_risk": "how the change could hurt other examples and how to contain it",\n'
        '  "operational_rule": "trigger -> evidence -> analysis -> transmission -> action",\n'
        '  "feedback": "concise actionable feedback",\n'
        '  "boundary_assessment": "whether the edit respects component roles"\n'
        "}\n\n"
        f"Allowed components: {list(candidate)}\n\n"
        f"Task: {example.get('input', '')}\n"
        f"Expected: {example.get('answer') or example.get('expected') or 'rubric-only'}\n"
        f"Rubric: {example.get('rubric') or 'n/a'}\n\n"
        f"Expert evaluation data:\n{example_data_text(example)}\n\n"
        f"Rubric checkpoints:\n{checkpoint_lines}\n"
        f"Rubric coverage by candidate output: {rubric_coverage:.2f}\n"
        f"Missing rubric checkpoints:\n{missing_checkpoint_lines}\n\n"
        f"Trace expectations:\n{trace_expectation_lines}\n"
        f"Trace expectation coverage by candidate trace: {trace_coverage:.2f}\n"
        f"Matched trace expectations:\n{matched_trace_expectation_lines}\n"
        f"Missing trace expectations:\n{missing_trace_expectation_lines}\n"
        f"Missing expectations with apparent tool support:\n{tool_supported_missing_lines}\n"
        f"Tool capability gaps:\n{tool_capability_gap_lines}\n"
        f"Deterministic mutation eligibility: {str(mutation_eligible).lower()}\n"
        f"Mutation eligibility reason: {mutation_eligibility_reason}\n"
        f"Successful tool evidence:\n{successful_tool_evidence_lines}\n"
        f"Failed tool evidence:\n{failed_tool_evidence_lines}\n"
        f"Available tools:\n{available_tools[:2500]}\n\n"
        f"Candidate output:\n{last_message_text(state)}\n\n"
        f"Baseline output:\n{state.get('baseline_response', '')}\n\n"
        f"Adaptive trace summary:\n{summarize_messages(state)}\n\n"
        f"Hard constraint failures:\n{hard_lines}\n\n"
        f"Advisory notes:\n{advisory_lines}\n\n"
        f"Deterministic fallback score: {deterministic_score:.3f}\n"
        f"Deterministic fallback feedback:\n{deterministic_feedback[:2000]}\n\n"
        "Candidate component excerpts:\n"
        f"{compact_candidate_excerpt(candidate)}"
    )


def compact_candidate_excerpt(candidate: dict[str, str], limit_per_component: int = 700) -> str:
    blocks = []
    for key, text in candidate.items():
        snippet = text if len(text) <= limit_per_component else text[:limit_per_component] + "\n...[truncated]"
        blocks.append(f"### {key}\n{snippet}")
    return "\n\n".join(blocks)


def parse_judge_json(raw_output: str) -> dict[str, Any] | None:
    text = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def coerce_score(value: Any, fallback: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = fallback
    return max(0.0, min(1.0, score))


def build_judge_feedback(
    *,
    example: dict[str, Any],
    state: dict,
    response: str,
    baseline_response: str,
    failures: list[dict[str, Any]],
    deterministic_feedback: str,
    raw_judge: str,
    payload: dict[str, Any],
    score: float,
    raw_score: float,
    cap: float,
    correctness_cap_value: float,
    rubric_cap_value: float,
    constraint_cap_value: float,
    failure_classification: str,
    classification_reason: str,
    suggested: str,
) -> str:
    failure_lines = "\n".join(f"- {f['name']}: {f['message']}" for f in failures[:20]) or "- none"
    suggested_reason = str(payload.get("suggested_component_reason") or "reflection judge recommendation")
    knowledge_scope = str(payload.get("knowledge_scope") or "not provided").strip()
    applicability_scope = str(payload.get("applicability_scope") or "not provided").strip()
    regression_risk = str(payload.get("cross_case_regression_risk") or "not provided").strip()
    operational_rule = str(payload.get("operational_rule") or "not provided").strip()
    reusable_lesson = str(payload.get("reusable_lesson") or "").strip()
    feedback_text = str(payload.get("feedback") or "").strip()
    boundary_assessment = str(payload.get("boundary_assessment") or "").strip()
    fitness = state.get("fitness", {})
    mutation_eligible = bool(fitness.get("mutation_eligible", True))
    mutation_eligibility_reason = str(fitness.get("mutation_eligibility_reason") or "not provided")
    if not mutation_eligible:
        suggested_reason = mutation_eligibility_reason
    matched_checkpoints, missing_checkpoints, rubric_coverage = rubric_checkpoint_results(example, response)
    matched_checkpoint_lines = "\n".join(f"- {item}" for item in matched_checkpoints) or "- none"
    missing_checkpoint_lines = "\n".join(f"- {item}" for item in missing_checkpoints) or "- none"
    acquisition_diagnostics = data_acquisition_diagnostics(example, state)
    matched_trace_expectations = acquisition_diagnostics["matched_trace_expectations"]
    missing_trace_expectations = acquisition_diagnostics["missing_trace_expectations"]
    trace_coverage = acquisition_diagnostics["trace_expectation_coverage"]
    matched_trace_expectation_lines = "\n".join(f"- {item}" for item in matched_trace_expectations) or "- none"
    missing_trace_expectation_lines = "\n".join(f"- {item}" for item in missing_trace_expectations) or "- none"
    tool_supported_missing_lines = (
        "\n".join(f"- {item}" for item in acquisition_diagnostics["tool_supported_missing_expectations"]) or "- none"
    )
    tool_capability_gap_lines = (
        "\n".join(f"- {item}" for item in acquisition_diagnostics["tool_capability_gaps"]) or "- none"
    )
    successful_tool_evidence_lines = tool_evidence_text(acquisition_diagnostics["successful_tool_evidence"])
    failed_tool_evidence_lines = tool_evidence_text(acquisition_diagnostics["failed_tool_evidence"])
    return (
        "Reflection-judge Deep Agents text-surface evaluation.\n"
        f"Task: {example.get('input', '')}\n"
        f"Expected: {example.get('answer') or example.get('expected') or 'rubric-only'}\n"
        f"Rubric: {example.get('rubric') or 'n/a'}\n"
        f"Expert evaluation data:\n{example_data_text(example, limit=1800)}\n\n"
        "Scores:\n"
        f"- judge_score: {raw_score:.2f}\n"
        f"- correctness_cap: {correctness_cap_value:.2f}\n"
        f"- rubric_cap: {rubric_cap_value:.2f}\n"
        f"- rubric_coverage: {rubric_coverage:.2f}\n"
        f"- trace_expectation_coverage: {trace_coverage:.2f}\n"
        f"- constraint_cap: {constraint_cap_value:.2f}\n"
        f"- final_cap: {cap:.2f}\n"
        f"- final_score: {score:.2f}\n"
        f"- eval_mode: llm_judge\n"
        f"- failure_classification: {failure_classification}\n"
        f"- classification_reason: {classification_reason}\n"
        f"- mutation_eligible: {str(mutation_eligible).lower()}\n"
        f"- mutation_eligibility_reason: {mutation_eligibility_reason}\n"
        f"- suggested_component: {suggested or 'none'}\n"
        f"- suggested_component_reason: {suggested_reason}\n"
        f"- reusable_lesson: {reusable_lesson or 'none'}\n"
        f"- knowledge_scope: {knowledge_scope}\n"
        f"- applicability_scope: {applicability_scope}\n"
        f"- cross_case_regression_risk: {regression_risk}\n"
        f"- operational_rule: {operational_rule}\n\n"
        "Gate failures:\n"
        f"{failure_lines}\n\n"
        "Matched rubric checkpoints:\n"
        f"{matched_checkpoint_lines}\n\n"
        "Missing rubric checkpoints:\n"
        f"{missing_checkpoint_lines}\n\n"
        "Matched trace expectations:\n"
        f"{matched_trace_expectation_lines}\n\n"
        "Missing trace expectations:\n"
        f"{missing_trace_expectation_lines}\n\n"
        "Missing expectations with apparent tool support:\n"
        f"{tool_supported_missing_lines}\n\n"
        "Tool capability gaps:\n"
        f"{tool_capability_gap_lines}\n\n"
        "Successful tool evidence:\n"
        f"{successful_tool_evidence_lines}\n\n"
        "Failed tool evidence:\n"
        f"{failed_tool_evidence_lines}\n\n"
        "Judge feedback:\n"
        f"{feedback_text or 'n/a'}\n\n"
        "Boundary assessment:\n"
        f"{boundary_assessment or 'n/a'}\n\n"
        "With candidate output:\n"
        f"{response}\n\n"
        "Baseline output:\n"
        f"{baseline_response}\n\n"
        "Adaptive trace summary:\n"
        f"{summarize_messages(state)}\n\n"
        "Deterministic fallback feedback:\n"
        f"{deterministic_feedback[:2000]}\n\n"
        "Raw judge output:\n"
        f"{raw_judge[:2000]}"
    )


def mutation_eligibility_diagnostics(
    example: dict[str, Any],
    state: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    fitness: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate behavior scoring from whether hidden evidence may drive a text mutation."""
    if failures:
        return {
            "mutation_eligible": True,
            "mutation_eligibility_reason": "candidate text itself violates a deterministic validity constraint",
            "mutation_block_classification": None,
        }

    expected = example.get("answer") or example.get("expected") or ""
    if expected:
        return {
            "mutation_eligible": True,
            "mutation_eligibility_reason": "the runtime output can be compared directly with an authoritative target",
            "mutation_block_classification": None,
        }

    supported_missing = list(fitness.get("tool_supported_missing_expectations") or [])
    if supported_missing:
        return {
            "mutation_eligible": True,
            "mutation_eligibility_reason": "the agent skipped available evidence-acquisition paths: "
            + ", ".join(str(item) for item in supported_missing[:3]),
            "mutation_block_classification": None,
        }

    missing_checkpoints = list(fitness.get("missing_rubric_checkpoints") or [])
    matched_expectations = list(fitness.get("matched_trace_expectations") or [])
    if missing_checkpoints and matched_expectations:
        return {
            "mutation_eligible": True,
            "mutation_eligibility_reason": "runtime evidence matched required acquisition paths: "
            + ", ".join(str(item) for item in matched_expectations[:3]),
            "mutation_block_classification": None,
        }

    tool_gaps = list(fitness.get("tool_capability_gaps") or [])
    if tool_gaps:
        return {
            "mutation_eligible": False,
            "mutation_eligibility_reason": "missing expert checkpoints depend on evidence unavailable from current tools: "
            + ", ".join(str(item) for item in tool_gaps[:3]),
            "mutation_block_classification": TOOL_CAPABILITY_GAP,
        }

    if not missing_checkpoints:
        return {
            "mutation_eligible": False,
            "mutation_eligibility_reason": "no missing expert checkpoint requires a text change",
            "mutation_block_classification": NO_FAILURE,
        }

    if trace_expectations(example):
        reason = "required runtime evidence was not established by a successful or supported tool path"
    else:
        reason = (
            "the evaluator-only opinion is not linked to runtime-observable evidence, so a reusable text lesson "
            "cannot be established safely"
        )
    return {
        "mutation_eligible": False,
        "mutation_eligibility_reason": reason,
        "mutation_block_classification": INSUFFICIENT_RUNTIME_EVIDENCE,
    }


def classify_failure(
    example: dict[str, Any],
    state: dict,
    response: str,
    failures: list[dict[str, Any]],
    fitness: dict[str, Any],
) -> tuple[str, str]:
    if failures:
        return SKILL_DEFECT, "candidate text failed one or more hard gates"

    tool_gaps = list(fitness.get("tool_capability_gaps") or [])
    supported_missing = list(fitness.get("tool_supported_missing_expectations") or [])

    expected = example.get("answer") or example.get("expected") or ""
    if not expected:
        _matched, missing, _coverage = rubric_checkpoint_results(example, response)
        if missing:
            if supported_missing:
                return (
                    EXECUTION_LAPSE,
                    "agent skipped available data-acquisition paths for: "
                    + ", ".join(str(item) for item in supported_missing[:3]),
                )
            if not bool(fitness.get("mutation_eligible", True)):
                blocked_classification = str(
                    fitness.get("mutation_block_classification") or INSUFFICIENT_RUNTIME_EVIDENCE
                )
                return blocked_classification, str(
                    fitness.get("mutation_eligibility_reason")
                    or "missing expert checkpoints are not supported by runtime-observable evidence"
                )
            reason = f"rubric-only output missed expert checkpoints: {', '.join(missing[:3])}"
            if tool_gaps:
                reason += "; separate tool capability gaps: " + ", ".join(str(item) for item in tool_gaps[:3])
            return SKILL_DEFECT, reason
        if supported_missing:
            return (
                EXECUTION_LAPSE,
                "agent skipped available data-acquisition paths for: "
                + ", ".join(str(item) for item in supported_missing[:3]),
            )
        if tool_gaps:
            return (
                TOOL_CAPABILITY_GAP,
                "current tools cannot obtain required evidence for: " + ", ".join(str(item) for item in tool_gaps[:3]),
            )
        if response.strip():
            return NO_FAILURE, "rubric-only output covered the expert checkpoints"
        return EXECUTION_LAPSE, "rubric-only example did not produce a usable answer"
    if tool_gaps:
        return (
            TOOL_CAPABILITY_GAP,
            "current tools cannot obtain required evidence for: " + ", ".join(str(item) for item in tool_gaps[:3]),
        )
    if float(fitness.get("hard", 0.0)) >= 1.0:
        return NO_FAILURE, "candidate produced the expected route"
    predicted = extract_route(response)
    if predicted is None:
        return EXECUTION_LAPSE, "agent did not follow the required <route> output contract"
    if candidate_mentions_expected_route(state, expected):
        return EXECUTION_LAPSE, "existing skill/reference text appears to cover this route, but execution missed it"
    return SKILL_DEFECT, "routing guidance appears missing, wrong, or underspecified for this case"


def candidate_mentions_expected_route(state: dict, expected_route: str) -> bool:
    needle = expected_route.lower()
    for key, text in state.get("candidate_excerpt", {}).items():
        if not isinstance(text, str):
            continue
        if (
            key.endswith(":SKILL.md")
            or ":reference/" in key
            or key.endswith(":system_prompt")
            or key.startswith("memory:")
        ) and needle in text.lower():
            return True
    return False


def build_feedback(
    example: dict[str, Any],
    state: dict,
    response: str,
    baseline_response: str,
    failures: list[dict[str, Any]],
    fitness: dict[str, Any],
) -> str:
    failure_lines = "\n".join(f"- {f['name']}: {f['message']}" for f in failures[:20]) or "- none"
    weakest = weakest_dimension(fitness, failures)
    failure_classification, classification_reason = classify_failure(example, state, response, failures, fitness)
    suggested = suggest_component_to_update(
        state,
        weakest,
        failure_classification,
        example.get("answer") or example.get("expected") or "",
    )
    suggested_reason = suggested_component_reason(failure_classification, suggested, weakest, failures)
    guidance = optimization_guidance(failure_classification, suggested, weakest, failures)
    matched_checkpoints = fitness.get("matched_rubric_checkpoints", [])
    missing_checkpoints = fitness.get("missing_rubric_checkpoints", [])
    matched_checkpoint_lines = "\n".join(f"- {item}" for item in matched_checkpoints) or "- none"
    missing_checkpoint_lines = "\n".join(f"- {item}" for item in missing_checkpoints) or "- none"
    matched_trace_expectations = fitness.get("matched_trace_expectations", [])
    missing_trace_expectations = fitness.get("missing_trace_expectations", [])
    matched_trace_expectation_lines = "\n".join(f"- {item}" for item in matched_trace_expectations) or "- none"
    missing_trace_expectation_lines = "\n".join(f"- {item}" for item in missing_trace_expectations) or "- none"
    tool_supported_missing_lines = (
        "\n".join(f"- {item}" for item in fitness.get("tool_supported_missing_expectations", [])) or "- none"
    )
    tool_capability_gap_lines = "\n".join(f"- {item}" for item in fitness.get("tool_capability_gaps", [])) or "- none"
    successful_tool_evidence_lines = tool_evidence_text(fitness.get("successful_tool_evidence", []))
    failed_tool_evidence_lines = tool_evidence_text(fitness.get("failed_tool_evidence", []))
    return (
        "Darwin-style Deep Agents text-surface evaluation.\n"
        f"Task: {example['input']}\n"
        f"Expected: {example.get('answer') or example.get('expected') or 'rubric-only'}\n"
        f"Rubric: {example.get('rubric') or 'n/a'}\n"
        f"Expert evaluation data:\n{example_data_text(example, limit=1800)}\n\n"
        "Scores:\n"
        f"- hard: {fitness['hard']:.2f}\n"
        f"- soft: {fitness['soft']:.2f}\n"
        f"- mixed: {fitness['mixed']:.2f}\n"
        f"- baseline_hard: {fitness['baseline_hard']:.2f}\n"
        f"- baseline_soft: {fitness['baseline_soft']:.2f}\n"
        f"- baseline_mixed: {fitness['baseline_mixed']:.2f}\n"
        f"- effect: {fitness['effect']:.2f}\n"
        f"- structure: {fitness['structure']:.2f}\n"
        f"- specificity: {fitness['specificity']:.2f}\n"
        f"- gate_penalty: {fitness['gate_penalty']:.2f}\n"
        f"- raw_composite: {fitness.get('raw_composite', fitness['composite']):.2f}\n"
        f"- correctness_cap: {fitness.get('correctness_cap', 1.0):.2f}\n"
        f"- rubric_cap: {fitness.get('rubric_cap', 1.0):.2f}\n"
        f"- rubric_coverage: {fitness.get('rubric_coverage', 1.0):.2f}\n"
        f"- trace_expectation_coverage: {fitness.get('trace_expectation_coverage', 1.0):.2f}\n"
        f"- constraint_cap: {fitness.get('constraint_cap', 1.0):.2f}\n"
        f"- eval_mode: {fitness['eval_mode']}\n"
        f"- weakest_dimension: {weakest}\n"
        f"- failure_classification: {failure_classification}\n"
        f"- classification_reason: {classification_reason}\n"
        f"- mutation_eligible: {str(bool(fitness.get('mutation_eligible', True))).lower()}\n"
        f"- mutation_eligibility_reason: {fitness.get('mutation_eligibility_reason', 'not provided')}\n"
        f"- suggested_component: {suggested or 'none'}\n"
        f"- suggested_component_reason: {suggested_reason}\n\n"
        "Gate failures:\n"
        f"{failure_lines}\n\n"
        "Matched rubric checkpoints:\n"
        f"{matched_checkpoint_lines}\n\n"
        "Missing rubric checkpoints:\n"
        f"{missing_checkpoint_lines}\n\n"
        "Matched trace expectations:\n"
        f"{matched_trace_expectation_lines}\n\n"
        "Missing trace expectations:\n"
        f"{missing_trace_expectation_lines}\n\n"
        "Missing expectations with apparent tool support:\n"
        f"{tool_supported_missing_lines}\n\n"
        "Tool capability gaps:\n"
        f"{tool_capability_gap_lines}\n\n"
        "Successful tool evidence:\n"
        f"{successful_tool_evidence_lines}\n\n"
        "Failed tool evidence:\n"
        f"{failed_tool_evidence_lines}\n\n"
        "Optimization guidance:\n"
        f"{guidance}\n\n"
        "With candidate output:\n"
        f"{response}\n\n"
        "Baseline output:\n"
        f"{baseline_response}\n\n"
        "Adaptive trace summary:\n"
        f"{summarize_messages(state)}"
    )


def optimization_guidance(
    failure_classification: str,
    suggested: str,
    weakest: str,
    failures: list[dict[str, Any]],
) -> str:
    if failures:
        boundary_failures = [failure for failure in failures if ":boundary:" in str(failure.get("name", ""))]
        if boundary_failures:
            return (
                f"- primary_action: repair `{suggested}` so it stays within its component boundary.\n"
                "- do_not: do not copy SKILL.md, reference files, AGENTS.md, tool descriptions, or candidate excerpt "
                "labels into another component.\n"
                f"- hard_gate: {boundary_failures[0]['name']} -> {boundary_failures[0]['message']}"
            )
        return (
            f"- primary_action: fix the hard gate on `{suggested}` before trying to improve behavior.\n"
            "- do_not: do not add more text until size, growth, runtime, and script-reference gates pass.\n"
            f"- weakest_dimension: {weakest}"
        )
    if failure_classification == EXECUTION_LAPSE:
        return (
            f"- primary_action: update `{suggested}` with a concise reminder to use existing skills/tools and obey the "
            "output contract.\n"
            "- do_not: do not duplicate task knowledge that already lives in SKILL.md or reference/*.md.\n"
            "- expected_change: make the agent call or consult existing resources more reliably."
        )
    if failure_classification in {TOOL_CAPABILITY_GAP, INSUFFICIENT_RUNTIME_EVIDENCE}:
        if failure_classification == INSUFFICIENT_RUNTIME_EVIDENCE:
            return (
                "- primary_action: do not mutate a text component from evaluator-only facts that are not connected to "
                "runtime-observable evidence.\n"
                "- framework_action: retain the sample for diagnosis and obtain a trace/tool-evidence mapping before "
                "promoting any lesson into a skill or reference.\n"
                "- do_not: do not memorize the expert answer, company facts, or exact checkpoint wording."
            )
        return (
            "- primary_action: do not mutate a text component to simulate unavailable data access.\n"
            "- framework_action: preserve the missing expectation as a TOOL_CAPABILITY_GAP artifact for tool backlog "
            "planning.\n"
            "- do_not: do not invent facts, tools, endpoints, or claim that a rewritten description changes tool code."
        )
    return (
        f"- primary_action: add or sharpen reusable task knowledge in `{suggested}`.\n"
        "- do_not: do not write a test-specific answer.\n"
        "- scope_requirement: state observable applicability signals and exclusions; do not turn one industry example "
        "into an unconditional global rule.\n"
        "- operational_requirement: encode a conditional trigger, borrower-specific evidence plan, analysis, risk "
        "transmission, and approval or verification action; adapt sources and comparisons to the business model, and "
        "avoid vague reminders.\n"
        "- expected_change: improve routing evidence, failure modes, or decision criteria."
    )


def weakest_dimension(fitness: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    if failures:
        return "gate"
    scored = {k: float(fitness[k]) for k in ["effect", "structure", "specificity"]}
    return min(scored, key=scored.get)


def suggest_component_to_update(
    state: dict,
    weakest: str,
    failure_classification: str = SKILL_DEFECT,
    expected_route: str = "",
) -> str:
    candidate = state.get("candidate_excerpt", {})
    if failure_classification in {TOOL_CAPABILITY_GAP, INSUFFICIENT_RUNTIME_EVIDENCE, NO_FAILURE}:
        return ""
    if failure_classification == EXECUTION_LAPSE:
        return prompt_or_memory_component(candidate)
    if failure_classification == SKILL_DEFECT and state.get("fitness", {}).get("missing_rubric_checkpoints"):
        weakest = "effect"

    failures = hard_constraint_failures(state)
    if failures:
        component = component_from_prefixed_name(str(failures[0]["name"]), candidate)
        if component is not None:
            return component
    if weakest == "effect":
        if expected_route:
            route = expected_route.lower()
            for key, text in candidate.items():
                if ":reference/" in key and route in text.lower():
                    return key
        component = first_component_matching(
            candidate,
            lambda key, _text: ":reference/" in key
            and any(marker in key.lower() for marker in ["learned_", "expert_pattern", "experience"]),
        )
        if component is not None:
            return component
        component = first_component_matching(candidate, lambda key, _text: key.endswith(":SKILL.md"))
        if component is not None:
            return component
        component = first_component_matching(candidate, lambda key, _text: ":tool:" in key)
        if component is not None:
            return component
    if weakest == "specificity":
        for key, text in candidate.items():
            if SOFTENER_PATTERN.search(text):
                return key
    return next(iter(candidate), "main:system_prompt")


def component_from_prefixed_name(name: str, candidate: dict[str, str]) -> str | None:
    if name in candidate:
        return name
    matches = [key for key in candidate if name.startswith(f"{key}:")]
    return max(matches, key=len) if matches else None


def prompt_or_memory_component(candidate: dict[str, str]) -> str:
    component = first_component_matching(candidate, lambda key, _text: key.startswith("memory:"))
    if component is not None:
        return component
    if "main:system_prompt" in candidate:
        return "main:system_prompt"
    component = first_component_matching(
        candidate, lambda key, _text: key.startswith("subagent:") and key.endswith(":system_prompt")
    )
    if component is not None:
        return component
    component = first_component_matching(
        candidate, lambda key, _text: key.startswith("subagent:") and key.endswith(":description")
    )
    if component is not None:
        return component
    return next(iter(candidate), "main:system_prompt")


def first_component_matching(candidate: dict[str, str], predicate: Callable[[str, str], bool]) -> str | None:
    for key, text in candidate.items():
        if predicate(key, text):
            return key
    return None


def suggested_component_reason(
    failure_classification: str,
    suggested: str,
    weakest: str,
    failures: list[dict[str, Any]],
) -> str:
    if failure_classification == TOOL_CAPABILITY_GAP:
        return "no current text component can add the missing external data capability"
    if failure_classification == INSUFFICIENT_RUNTIME_EVIDENCE:
        return "evaluator-only facts are not connected to evidence observable by the runtime agent"
    if failure_classification == EXECUTION_LAPSE:
        return f"{suggested} can remind the agent to use already-available skills, references, and tools"
    if failures:
        return "the component is attached to the first hard-gate failure"
    if weakest == "effect":
        return "task output is weak, so update the most relevant task knowledge surface"
    if weakest == "specificity":
        return "component contains softened language that can be made more actionable"
    return "fallback to the first available candidate component"


def reflective_record(example: dict[str, Any], state: dict, score: float, feedback: str) -> dict[str, Any]:
    response = last_message_text(state)
    baseline_response = state.get("baseline_response", "")
    fitness = {
        key: value
        for key, value in dict(state.get("fitness", {})).items()
        if key not in {"successful_tool_evidence", "failed_tool_evidence", "trace_expectation_evidence"}
    }
    return {
        "Runtime input": example["input"],
        "Expected": example.get("answer") or example.get("expected"),
        "Rubric": example.get("rubric"),
        "Evaluator-only expert evidence (never shown to runtime agent)": example.get("data"),
        "Agent response": response,
        "Baseline response": "[same as Agent response]" if baseline_response == response else baseline_response,
        "Score": score,
        "Fitness": fitness,
        "Feedback": compact_feedback_for_reflection(feedback),
        "Failed constraints": [
            constraint
            for constraint in state.get("candidate_constraints", [])
            if isinstance(constraint, dict) and not constraint.get("passed", True)
        ],
        "Changed candidate metrics": compact_candidate_metrics_for_reflection(state.get("candidate_metrics", {})),
        "Recent trace": summarize_messages(state),
        "Project component map": compact_candidate_map_for_reflection(state.get("candidate_excerpt", {})),
    }


def compact_candidate_metrics_for_reflection(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        return {}
    growth = metrics.get("growth")
    lengths = metrics.get("lengths")
    if not isinstance(growth, Mapping):
        return {}
    changed_growth = {
        str(key): float(value)
        for key, value in growth.items()
        if isinstance(value, int | float) and abs(float(value)) > 1e-12
    }
    changed_lengths = (
        {key: lengths[key] for key in changed_growth if isinstance(lengths, Mapping) and key in lengths}
        if changed_growth
        else {}
    )
    return {"lengths": changed_lengths, "growth": changed_growth} if changed_growth else {}


def compact_candidate_map_for_reflection(candidate: Any, limit_per_component: int = 500) -> dict[str, str]:
    if not isinstance(candidate, Mapping):
        return {}
    compact: dict[str, str] = {}
    for key, value in candidate.items():
        text = str(value)
        compact[str(key)] = (
            text
            if len(text) <= limit_per_component
            else text[:limit_per_component] + "\n...[component excerpt truncated]"
        )
    return compact


def compact_feedback_for_reflection(feedback: str) -> str:
    """Drop sections already represented as structured reflection-record fields."""
    duplicate_section_markers = (
        "\nWith candidate output:\n",
        "\nBaseline output:\n",
        "\nAdaptive trace summary:\n",
        "\nDeterministic fallback feedback:\n",
        "\nRaw judge output:\n",
    )
    cutoffs = [feedback.find(marker) for marker in duplicate_section_markers if marker in feedback]
    return feedback[: min(cutoffs)].rstrip() if cutoffs else feedback


def log_agent_evaluation(
    example: dict[str, Any],
    state: dict[str, Any],
    score: float,
    feedback: str,
    artifact_store: RunArtifactStore | None,
) -> None:
    input_preview = str(example.get("input", ""))[:200].replace("\n", " ")
    response_preview = last_message_text(state)[:300].replace("\n", " ")
    fitness = state.get("fitness", {})
    LOGGER.info(
        "agent_eval score=%.3f candidate=%s hard=%s effect=%s input=%s response=%s",
        score,
        state.get("candidate_hash", "unknown"),
        fitness.get("hard"),
        fitness.get("effect"),
        input_preview,
        response_preview,
    )
    LOGGER.debug("agent_eval feedback:\n%s", feedback)
    LOGGER.debug("agent_eval recent trace:\n%s", summarize_messages(state))
    if artifact_store is not None:
        artifact_store.write_agent_rollout(example=example, state=state, score=score, feedback=feedback)


def with_rejected_history(
    reflection_callable: Callable[[str], str],
    history_block: Callable[[], str],
) -> Callable[[str], str]:
    def reflection_lm(prompt: str) -> str:
        block = history_block()
        if block:
            prompt = f"{prompt}\n\n{block}\n"
        return reflection_callable(prompt)

    return reflection_lm


def current_component_from_reflection_prompt(prompt: str) -> str | None:
    match = re.search(
        r"(?s)Current target component \(this is the only text you may replace\):\n```\n(.*?)\n```\n\n"
        r"Before answering,",
        prompt,
    )
    return match.group(1) if match else None


def no_change_proposal_response(prompt: str, issues: Sequence[str]) -> str | None:
    current_component = current_component_from_reflection_prompt(prompt)
    if current_component is None:
        return None
    issue_text = "; ".join(str(issue) for issue in issues if str(issue).strip()) or (
        "available optimizer evidence does not justify a runtime text mutation"
    )
    return (
        "Proposal rationale:\n"
        f"- Failure pattern: {issue_text}\n"
        "- Evidence across examples: no runtime-observable evidence supports promoting hidden case facts.\n"
        "- Selected component: no change.\n"
        "- Why this component: the selected component is preserved exactly.\n"
        "- Why not other components: the gap belongs to evidence/tooling, not another text surface.\n"
        "- Applicability scope and exclusions: n/a.\n"
        "- Cross-case regression risk: avoided by retaining the incumbent text.\n"
        "- Operational rule shape: no new rule.\n"
        "- Boundary checks: unchanged component remains within its boundary.\n"
        "- Hidden-data boundary check: evaluator-only facts were not persisted.\n"
        "- Intended behavior change: none until runtime evidence or tooling makes the lesson learnable.\n\n"
        "Final replacement:\n"
        "```markdown\n"
        f"{current_component}\n"
        "```"
    )


def with_proposal_quality_review(
    proposal_callable: Callable[[str], str],
    review_callable: Callable[[str], str],
    reviewer: ProposalReviewer,
    artifact_store: RunArtifactStore | None = None,
) -> Callable[[str], str]:
    """Review and optionally revise each reflected proposal before candidate rollout."""

    def reflection_lm(prompt: str) -> str:
        original_response = proposal_callable(prompt)
        try:
            review = reviewer.review(
                reflection_prompt=prompt,
                proposal_response=original_response,
                review_lm=review_callable,
            )
        except Exception as exc:  # pragma: no cover - provider-specific defensive fallback.
            if artifact_store is not None:
                artifact_store.write_proposal_review(
                    prompt=prompt,
                    original_response=original_response,
                    decision="REVIEW_ERROR",
                    issues=(),
                    raw_review="",
                    reviewed_response=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            LOGGER.warning("proposal review failed; retaining original proposal: %s: %s", type(exc).__name__, exc)
            return original_response

        reviewed_response = original_response
        if review.decision == "REVISE" and review.reviewed_response is not None:
            reviewed_response = review.reviewed_response
        elif review.decision == "REJECT":
            reviewed_response = no_change_proposal_response(prompt, review.issues) or original_response

        if artifact_store is not None:
            artifact_store.write_proposal_review(
                prompt=prompt,
                original_response=original_response,
                decision=review.decision,
                issues=review.issues,
                raw_review=review.raw_output,
                reviewed_response=reviewed_response if reviewed_response != original_response else None,
            )
        LOGGER.info(
            "proposal_review decision=%s issues=%d original_chars=%d reviewed_chars=%d",
            review.decision,
            len(review.issues),
            len(original_response),
            len(reviewed_response),
        )
        return reviewed_response

    return reflection_lm


def with_reflection_error_artifacts(
    reflection_callable: Callable[[str], str],
    artifact_store: RunArtifactStore,
) -> Callable[[str], str]:
    def reflection_lm(prompt: str) -> str:
        try:
            return reflection_callable(prompt)
        except Exception as exc:
            artifact_store.write_reflection_error(prompt=prompt, error=exc)
            raise

    return reflection_lm


def generate_dataset() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rows = [
        ("I was charged twice for my invoice this month.", "billing"),
        ("Please reset my password; I cannot sign in.", "account"),
        ("The export button crashes every time I click it.", "engineering"),
        ("Can you add a Salesforce integration next quarter?", "product"),
        ("Where can I download my receipt?", "billing"),
        ("My two-factor authentication code is not accepted.", "account"),
        ("The dashboard shows a 500 error.", "engineering"),
        ("I want an integration with Notion.", "product"),
    ]
    data = [{"input": text, "answer": answer} for text, answer in rows]
    return data[:5], data[5:7], data[7:]


def load_dataset_from_config(
    config: DeepAgentsGepaConfig,
    langfuse_client: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build train/val/test examples from a configured dataset source."""
    source = config.dataset.source
    if source == "synthetic":
        return generate_dataset()
    if source in {"golden", "golden_jsonl", "jsonl"}:
        records = load_golden_jsonl(config)
    elif source in {"langfuse_experience", "langfuse_labeled"}:
        records = load_langfuse_records(config, langfuse_client)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")
    examples = [record.as_example() for record in records]
    if config.dataset.limit is not None:
        examples = examples[: config.dataset.limit]
    return split_examples(
        examples,
        split_strategy=config.dataset.split_strategy,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
        stratify_by=config.dataset.stratify_by,
        seed=config.dataset.seed,
    )


def load_golden_jsonl(config: DeepAgentsGepaConfig) -> list[EvalRecord]:
    if config.dataset.path is None:
        raise ValueError("golden_jsonl dataset requires dataset.path")
    path = _resolve_source_path(config.project_root, config.dataset.path)
    records: list[EvalRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(normalize_eval_record(payload, {"source": "golden_jsonl", "line": line_no}, path.parent))
    return records


def normalize_eval_record(
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    base_path: Path | None = None,
) -> EvalRecord:
    merged_metadata = dict(metadata or {})
    merged_metadata.update(payload.get("metadata", {}))
    if payload.get("split") is not None:
        merged_metadata["split"] = str(payload["split"])
    if payload.get("stratum") is not None:
        merged_metadata["stratum"] = str(payload["stratum"])
    messages = tuple(_normalize_message(message) for message in payload.get("messages", []))
    data = payload.get("data") or payload.get("expert_risk_section") or payload.get("expert_opinion")
    data_path = (
        payload.get("data_path") or payload.get("expert_risk_section_path") or payload.get("expert_opinion_path")
    )
    if data is None and data_path is not None:
        resolved_data_path = Path(_expand_value(str(data_path)))
        if not resolved_data_path.is_absolute():
            resolved_data_path = (base_path or Path.cwd()) / resolved_data_path
        data = resolved_data_path.read_text(encoding="utf-8")
        merged_metadata.setdefault("data_path", str(data_path))
    return EvalRecord(
        input=payload.get("input"),
        data=str(data) if data is not None else None,
        messages=messages,
        expected=payload.get("expected") or payload.get("answer"),
        rubric=payload.get("rubric"),
        metadata=merged_metadata,
    )


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    return {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}


def load_langfuse_records(config: DeepAgentsGepaConfig, langfuse_client: Any | None = None) -> list[EvalRecord]:
    traces = fetch_langfuse_traces(config, langfuse_client)
    labeled_only = config.dataset.source == "langfuse_labeled"
    records = records_from_langfuse_traces(traces, labeled_only=labeled_only)
    if config.dataset.limit is not None:
        return records[: config.dataset.limit]
    return records


def fetch_langfuse_traces(config: DeepAgentsGepaConfig, langfuse_client: Any | None) -> list[dict[str, Any]]:
    """Fetch traces through an injected client, keeping SDK specifics isolated."""
    if langfuse_client is None:
        raise ValueError("Langfuse dataset loading requires an injected langfuse_client")
    query = dict(config.dataset.query)
    if hasattr(langfuse_client, "fetch_traces"):
        return list(langfuse_client.fetch_traces(**query))
    if hasattr(langfuse_client, "get_traces"):
        return list(langfuse_client.get_traces(**query))
    if isinstance(langfuse_client, Sequence) and not isinstance(langfuse_client, str | bytes):
        return [dict(trace) for trace in langfuse_client]
    raise TypeError("langfuse_client must expose fetch_traces/get_traces or be a trace sequence")


def records_from_langfuse_traces(traces: Sequence[dict[str, Any]], labeled_only: bool = False) -> list[EvalRecord]:
    """Convert online conversation traces into optimization examples.

    Unlabeled traces are experience-mining data: user questions, corrections,
    and follow-ups are valuable even when the assistant's final answer is not
    trusted. Labeled mode keeps only traces with explicit score/feedback or an
    accepted expected output.
    """
    records: list[EvalRecord] = []
    for trace in traces:
        messages = extract_trace_messages(trace)
        expected = extract_trace_expected(trace)
        rubric = trace.get("rubric") or build_trace_rubric(trace, messages)
        has_label = expected is not None or trace.get("score") is not None or trace.get("user_feedback") is not None
        if labeled_only and not has_label:
            continue
        metadata = {
            "source": "langfuse",
            "trace_id": trace.get("id") or trace.get("trace_id"),
            "score": trace.get("score"),
            "user_feedback": trace.get("user_feedback"),
            "labeled": has_label,
        }
        if labeled_only:
            records.append(EvalRecord(messages=tuple(messages), expected=expected, rubric=rubric, metadata=metadata))
            continue
        for index, message in enumerate(messages):
            if message["role"] != "user" or not message["content"].strip():
                continue
            records.append(
                EvalRecord(
                    input=message["content"],
                    rubric=rubric,
                    metadata={**metadata, "turn_index": index, "experience_kind": classify_user_experience(message)},
                )
            )
    return records


def extract_trace_messages(trace: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = trace.get("messages")
    if raw_messages is None and isinstance(trace.get("input"), list):
        raw_messages = trace["input"]
    if raw_messages is None:
        raw_messages = []
        if trace.get("input"):
            raw_messages.append({"role": "user", "content": str(trace["input"])})
        if trace.get("output"):
            raw_messages.append({"role": "assistant", "content": str(trace["output"])})
    return [_normalize_message(message) for message in raw_messages]


def extract_trace_expected(trace: dict[str, Any]) -> str | None:
    if trace.get("expected") is not None:
        return str(trace["expected"])
    if trace.get("accepted_output") is not None:
        return str(trace["accepted_output"])
    return None


def build_trace_rubric(trace: dict[str, Any], messages: Sequence[dict[str, str]]) -> str:
    user_turns = [message["content"] for message in messages if message["role"] == "user"]
    focus = "; ".join(user_turns[-3:]) if user_turns else "the user's task"
    feedback = trace.get("user_feedback")
    if feedback:
        return f"Satisfy the user's latest intent and address this feedback: {feedback}"
    return f"Satisfy the expert user's intent, including follow-up corrections and risk checks: {focus}"


def classify_user_experience(message: dict[str, str]) -> str:
    text = message["content"].lower()
    if any(word in text for word in ["wrong", "incorrect", "不对", "错", "纠正"]):
        return "correction"
    if any(word in text for word in ["risk", "风险", "verify", "验证", "check", "核查"]):
        return "risk_probe"
    if "?" in text:
        return "follow_up_question"
    return "expert_task"


def split_examples(
    examples: list[dict[str, Any]],
    *,
    split_strategy: str = "stratified",
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    stratify_by: Sequence[str] = ("metadata.difficulty",),
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], [], []
    if len(examples) == 1:
        only = tag_dataset_split(examples[0], "train", stratum=dataset_stratum(examples[0], stratify_by))
        validation_copy = tag_dataset_split(
            examples[0],
            "val",
            stratum=dataset_stratum(examples[0], stratify_by),
            fallback="single_example_reused_for_validation",
        )
        return [only], [validation_copy], []

    split_names = ("train", "val", "test")
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    unassigned: list[dict[str, Any]] = []
    for example in examples:
        metadata = example.get("metadata") if isinstance(example.get("metadata"), Mapping) else {}
        explicit_split = str(metadata.get("split") or metadata.get("dataset_split") or "").strip().lower()
        if explicit_split in buckets:
            buckets[explicit_split].append(
                tag_dataset_split(example, explicit_split, stratum=dataset_stratum(example, stratify_by))
            )
        else:
            unassigned.append(example)
    if split_strategy == "explicit" and unassigned:
        raise ValueError("dataset split_strategy=explicit requires every example to declare split=train|val|test")

    targets = dataset_split_targets(len(examples), train_ratio, val_ratio, test_ratio)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in unassigned:
        grouped.setdefault(dataset_stratum(example, stratify_by), []).append(example)
    ordered_groups = sorted(grouped.items(), key=lambda item: stable_split_key(item[0], seed))
    for stratum, rows in ordered_groups:
        for example in sorted(rows, key=lambda row: stable_split_key(str(row.get("input", "")), seed)):
            destination = max(
                split_names,
                key=lambda name: (
                    targets[name] - len(buckets[name]),
                    targets[name],
                    -split_names.index(name),
                ),
            )
            buckets[destination].append(tag_dataset_split(example, destination, stratum=stratum))

    if not buckets["train"]:
        buckets["train"].append(buckets["val"].pop() if buckets["val"] else buckets["test"].pop())
    if not buckets["val"]:
        source = buckets["train"] if len(buckets["train"]) > 1 else buckets["test"]
        if source:
            moved = source.pop()
            buckets["val"].append(tag_dataset_split(moved, "val", stratum=dataset_stratum(moved, stratify_by)))
    return buckets["train"], buckets["val"], buckets["test"]


def dataset_split_targets(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    ratios = [max(0.0, float(value)) for value in (train_ratio, val_ratio, test_ratio)]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("dataset split ratios must contain at least one positive value")
    normalized = [value / ratio_sum for value in ratios]
    if total == 2:
        return {"train": 1, "val": 1, "test": 0}
    val_count = max(1, round(total * normalized[1]))
    test_count = max(1, round(total * normalized[2]))
    train_count = total - val_count - test_count
    while train_count < 1:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
        train_count = total - val_count - test_count
    return {"train": train_count, "val": val_count, "test": test_count}


def dataset_stratum(example: Mapping[str, Any], stratify_by: Sequence[str]) -> str:
    values = []
    for path in stratify_by:
        value: Any = example
        for part in str(path).split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value not in (None, "", [], {}):
            values.append(f"{path}={value}")
    metadata = example.get("metadata") if isinstance(example.get("metadata"), Mapping) else {}
    if metadata.get("stratum"):
        values.insert(0, f"metadata.stratum={metadata['stratum']}")
    return "|".join(values) or "__all__"


def stable_split_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def tag_dataset_split(
    example: Mapping[str, Any],
    split: str,
    *,
    stratum: str,
    fallback: str | None = None,
) -> dict[str, Any]:
    tagged = dict(example)
    metadata = dict(example.get("metadata") or {})
    metadata["dataset_split"] = split
    metadata["dataset_stratum"] = stratum
    if fallback:
        metadata["dataset_split_fallback"] = fallback
    tagged["metadata"] = metadata
    return tagged


def examples_for_evaluation_phase(
    examples: Sequence[Mapping[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    return [{**dict(example), "evaluation_phase": phase} for example in examples]


def final_test_summary(seed_evaluation: Any, best_evaluation: Any) -> dict[str, Any]:
    seed_scores = [float(score) for score in list(getattr(seed_evaluation, "scores", []) or [])]
    best_scores = [float(score) for score in list(getattr(best_evaluation, "scores", []) or [])]
    seed_mean = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
    best_mean = sum(best_scores) / len(best_scores) if best_scores else 0.0
    return {
        "count": max(len(seed_scores), len(best_scores)),
        "seed_mean": seed_mean,
        "best_mean": best_mean,
        "improvement": best_mean - seed_mean,
        "per_example": [
            {
                "seed_score": seed_score,
                "best_score": best_score,
                "delta": best_score - seed_score,
            }
            for seed_score, best_score in zip(seed_scores, best_scores, strict=False)
        ],
    }


def run_configured_skill_optimization(
    config_path: str | Path,
    task_llm: BaseChatModel,
    reflection_llm: BaseChatModel | Callable[[str], str],
    *,
    tool_registry: dict[str, BaseTool | Callable | dict[str, Any]] | None = None,
    langfuse_client: Any | None = None,
    dataset_provider: DatasetProvider | None = None,
    evaluator: Evaluator | Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[float, str]] | None = None,
    template_registry: ReflectionTemplateRegistry | None = None,
    component_selector: ComponentSelector | None = None,
    proposal_reviewer: ProposalReviewer | None = None,
    constraint_policy: Constraint | None = None,
    mcp_loader: Callable[[Sequence[MCPServerConfig], dict[str, str]], Sequence[BaseTool | Callable | dict[str, Any]]]
    | None = None,
    max_metric_calls: int = 10,
    reflection_minibatch_size: int = 3,
    num_threads: int = 1,
    seed: int = 0,
    artifact_dir: str | Path | None = None,
    artifact_run_name: str | None = None,
    use_reflection_judge: bool = True,
    review_proposals: bool = True,
    evaluate_final_test: bool | None = None,
    evaluate_tied_candidates: bool = True,
) -> Any:
    """End-to-end config-driven optimization entry point."""
    config = load_deepagents_gepa_config(config_path)
    project = build_candidate_from_deep_agent_project(config, tool_registry=tool_registry)
    provider = dataset_provider or DefaultDatasetProvider(
        config, load_dataset_from_config, langfuse_client=langfuse_client
    )
    train_set, val_set, test_set = provider.load()
    if not train_set:
        raise ValueError("Configured dataset produced no training examples")
    artifact_store = RunArtifactStore.create(artifact_dir, artifact_run_name) if artifact_dir is not None else None
    if artifact_store is not None:
        artifact_store.write_run_inputs(
            config_path=config_path,
            config=config,
            project=project,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
        )
    seed_candidate = project.candidate
    templates = (template_registry or DefaultReflectionTemplateRegistry()).templates_for(seed_candidate)
    artifact_callback = artifact_store.create_callback() if artifact_store is not None else None
    base_reflection_callable = reflection_llm if callable(reflection_llm) else make_reflection_lm(reflection_llm)
    if evaluator is not None:
        active_evaluator = evaluator if hasattr(evaluator, "evaluate") else DefaultEvaluator(evaluator)  # type: ignore[arg-type]
    elif use_reflection_judge:
        active_evaluator = DefaultEvaluator(
            lambda example, state: evaluate_response_with_judge(example, state, base_reflection_callable)
        )
    else:
        active_evaluator = DefaultEvaluator(evaluate_response)

    def evaluate_and_log(example: dict[str, Any], state: dict[str, Any]) -> tuple[float, str]:
        prepare_evaluation_trace(state, base_reflection_callable)
        score, feedback = active_evaluator.evaluate(example, state)  # type: ignore[union-attr]
        log_agent_evaluation(example, state, score, feedback, artifact_store)
        return score, feedback

    reflection_callable = base_reflection_callable
    if review_proposals:
        reflection_callable = with_proposal_quality_review(
            reflection_callable,
            base_reflection_callable,
            proposal_reviewer or DefaultProposalReviewer(),
            artifact_store,
        )
    if artifact_callback is not None:
        reflection_callable = with_rejected_history(
            reflection_callable, artifact_callback.rejected_history_prompt_block
        )
    if artifact_store is not None:
        reflection_callable = with_reflection_error_artifacts(reflection_callable, artifact_store)
    adapter = LangChainAdapter(
        rollout_fn=lambda candidate, example: configured_rollout(
            candidate,
            example,
            task_llm,
            project,
            seed_candidate,
            mcp_loader,
            constraint_policy,
        ),
        eval_fn=evaluate_and_log,
        reflective_record_fn=reflective_record,
        num_threads=num_threads,
    )
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        reflection_lm=reflection_callable,
        reflection_prompt_template=templates,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        module_selector=component_selector or DefaultFeedbackComponentSelector(),
        candidate_selection_strategy="pareto",
        use_merge=True,
        display_progress_bar=False,
        callbacks=[artifact_callback] if artifact_callback is not None else None,
        seed=seed,
    )
    deployment_best_idx = select_deployment_candidate_index(result)
    deployment_candidate = (
        result.candidates[deployment_best_idx] if deployment_best_idx is not None else result.best_candidate
    )
    if deployment_best_idx is not None and deployment_best_idx != result.best_idx:
        LOGGER.info(
            "deployment candidate selection chose candidate=%d instead of gepa_best=%d at val_score=%.3f",
            deployment_best_idx,
            result.best_idx,
            result.val_aggregate_scores[deployment_best_idx],
        )
    final_test_result: dict[str, Any] | None = None
    should_evaluate_test = config.dataset.evaluate_final_test if evaluate_final_test is None else evaluate_final_test
    if should_evaluate_test and test_set:
        seed_test = adapter.evaluate(
            examples_for_evaluation_phase(test_set, "final_test_seed"),
            seed_candidate,
            capture_traces=False,
        )
        if deployment_candidate == seed_candidate:
            best_test = seed_test
        else:
            best_test = adapter.evaluate(
                examples_for_evaluation_phase(test_set, "final_test_best"),
                deployment_candidate,
                capture_traces=False,
            )
        diagnostic_evaluations: dict[int, Any] = {}
        diagnostic_val_scores: dict[int, float] = {}
        if evaluate_tied_candidates and deployment_best_idx is not None:
            val_scores = [float(score) for score in list(getattr(result, "val_aggregate_scores", []) or [])]
            if val_scores:
                top_score = max(val_scores)
                for candidate_idx, candidate in enumerate(result.candidates):
                    if candidate_idx == deployment_best_idx or abs(val_scores[candidate_idx] - top_score) > 1e-12:
                        continue
                    diagnostic_evaluations[candidate_idx] = adapter.evaluate(
                        examples_for_evaluation_phase(test_set, f"final_test_tied_candidate_{candidate_idx}"),
                        candidate,
                        capture_traces=False,
                    )
                    diagnostic_val_scores[candidate_idx] = val_scores[candidate_idx]
        final_test_result = final_test_summary(seed_test, best_test)
        if diagnostic_evaluations:
            seed_mean = final_test_result["seed_mean"]
            final_test_result["diagnostic_candidates"] = [
                {
                    "candidate_idx": candidate_idx,
                    "validation_score": diagnostic_val_scores[candidate_idx],
                    "test_mean": (
                        sum(float(score) for score in evaluation.scores) / len(evaluation.scores)
                        if evaluation.scores
                        else 0.0
                    ),
                    "delta_vs_seed": (
                        sum(float(score) for score in evaluation.scores) / len(evaluation.scores) - seed_mean
                        if evaluation.scores
                        else -seed_mean
                    ),
                    "selection_effect": "diagnostic_only",
                }
                for candidate_idx, evaluation in sorted(diagnostic_evaluations.items())
            ]
        if artifact_store is not None:
            final_test_result = artifact_store.write_final_test(
                examples=test_set,
                seed_evaluation=seed_test,
                best_evaluation=best_test,
                diagnostic_evaluations=diagnostic_evaluations,
                diagnostic_val_scores=diagnostic_val_scores,
            )
        LOGGER.info(
            "final_test count=%d seed_mean=%.3f best_mean=%.3f improvement=%.3f",
            final_test_result["count"],
            final_test_result["seed_mean"],
            final_test_result["best_mean"],
            final_test_result["improvement"],
        )
    elif should_evaluate_test:
        LOGGER.warning("final_test skipped because the configured dataset has no held-out test examples")
    if artifact_store is not None:
        artifact_store.finalize(
            result=result,
            project=project,
            apply_candidate=apply_candidate_to_deep_agent_project,
            final_test=final_test_result,
            best_idx=deployment_best_idx,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Optional deepagents_gepa.toml config path.")
    parser.add_argument("--task-model", default="openai:gpt-4o-mini")
    parser.add_argument("--task-model-kwargs", type=json.loads, default={})
    parser.add_argument("--reflection-model", default="openai:gpt-5-mini")
    parser.add_argument("--reflection-model-kwargs", type=json.loads, default={"reasoning_effort": "medium"})
    parser.add_argument("--max-metric-calls", type=int, default=50)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-dir", help="Optional base directory for run artifacts.")
    parser.add_argument("--artifact-run-name", help="Optional run directory name under --artifact-dir.")
    parser.add_argument(
        "--no-reflection-judge", action="store_true", help="Use deterministic eval instead of LLM judge."
    )
    parser.add_argument("--skip-proposal-review", action="store_true", help="Skip the pre-runtime LLM proposal review.")
    parser.add_argument(
        "--skip-final-test", action="store_true", help="Do not evaluate seed and best on held-out test."
    )
    parser.add_argument(
        "--skip-tied-candidate-test",
        action="store_true",
        help="Do not evaluate non-deployed validation-tied candidates on held-out data for diagnostics.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-optimize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_llm = init_chat_model(args.task_model, **args.task_model_kwargs)
    reflection_llm = init_chat_model(args.reflection_model, **args.reflection_model_kwargs)

    if args.config:
        result = run_configured_skill_optimization(
            args.config,
            task_llm,
            reflection_llm,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            num_threads=args.num_threads,
            seed=args.seed,
            artifact_dir=args.artifact_dir,
            artifact_run_name=args.artifact_run_name,
            use_reflection_judge=not args.no_reflection_judge,
            review_proposals=not args.skip_proposal_review,
            evaluate_final_test=not args.skip_final_test,
            evaluate_tied_candidates=not args.skip_tied_candidate_test,
        )
        print(f"\nBest val score: {result.val_aggregate_scores[result.best_idx]}")
        print("\nOptimized components:")
        for name, text in result.best_candidate.items():
            print(f"\n--- {name} ---\n{text}")
        return

    train_set, val_set, test_set = generate_dataset()

    with tempfile.TemporaryDirectory(prefix="gepa_deep_agent_seed_") as seed_tmp:
        seed_spec = create_seed_workspace(Path(seed_tmp))
        seed_candidate, surfaces = build_candidate_from_deep_agent_spec(seed_spec)
        templates = reflection_prompt_templates(seed_candidate)

        adapter = LangChainAdapter(
            rollout_fn=lambda candidate, example: rollout(
                candidate,
                example,
                task_llm,
                seed_spec,
                surfaces,
                seed_candidate,
            ),
            eval_fn=evaluate_response,
            reflective_record_fn=reflective_record,
            num_threads=args.num_threads,
        )

        if not args.skip_baseline:
            print("\nBaseline evaluation on test set...")
            baseline = adapter.evaluate(test_set, seed_candidate, capture_traces=False)
            print(f"Baseline score: {sum(baseline.scores):.3f}/{len(baseline.scores)}")

        if args.skip_optimize:
            return

        result = optimize(
            seed_candidate=seed_candidate,
            trainset=train_set,
            valset=val_set,
            adapter=adapter,
            reflection_lm=make_reflection_lm(reflection_llm),
            reflection_prompt_template=templates,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            module_selector=DarwinFeedbackComponentSelector(),
            candidate_selection_strategy="pareto",
            use_merge=True,
            display_progress_bar=True,
            seed=args.seed,
        )

        print(f"\nBest val score: {result.val_aggregate_scores[result.best_idx]}")
        print("\nOptimized components:")
        for name, text in result.best_candidate.items():
            print(f"\n--- {name} ---\n{text}")

        print("\nOptimized evaluation on test set...")
        optimized = adapter.evaluate(test_set, result.best_candidate, capture_traces=False)
        print(f"Optimized score: {sum(optimized.scores):.3f}/{len(optimized.scores)}")


if __name__ == "__main__":
    main()
