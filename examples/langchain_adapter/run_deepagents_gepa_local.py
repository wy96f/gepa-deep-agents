"""Run the Deep Agents GEPA example against a local OpenAI-compatible model.

This runner is intentionally tiny so it is easy to run or debug from PyCharm.
It creates LangChain ChatOpenAI clients with proxy auto-detection disabled, then
calls the config-driven Deep Agents GEPA harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_openai import ChatOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.langchain_adapter.deep_agent_skill_directory import (
    run_configured_skill_optimization,
    select_deployment_candidate_index,
)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_ENV_VARS = ("NO_PROXY", "no_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a Deep Agents GEPA TOML config. Use langgraph_cli.toml for LangGraph CLI auto-discovery.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="local-chat-model")
    parser.add_argument("--api-key", default="no-key")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--task-max-tokens", type=int, default=64 * 1024)
    parser.add_argument("--reflection-max-tokens", type=int, default=128 * 1024)
    parser.add_argument("--context-window-tokens", type=int, default=200_000)
    parser.add_argument("--trace-context-ratio", type=float, default=0.12)
    parser.add_argument("--trace-keep-ratio", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--max-metric-calls", type=int, default=2)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-reflection-judge", action="store_true", help="Use deterministic eval instead of LLM judge."
    )
    parser.add_argument("--skip-proposal-review", action="store_true", help="Skip the pre-runtime LLM proposal review.")
    parser.add_argument("--skip-final-test", action="store_true", help="Skip held-out seed/best test evaluation.")
    parser.add_argument(
        "--skip-tied-candidate-test",
        action="store_true",
        help="Do not evaluate non-deployed validation-tied candidates on held-out data for diagnostics.",
    )
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-file", help="Optional file path for detailed Python logs.")
    parser.add_argument("--summary-file", help="Optional JSON file path for a compact GEPA run summary.")
    parser.add_argument(
        "--artifact-dir",
        default=str(REPO_ROOT / "examples/langchain_adapter/runs"),
        help="Base directory for run artifacts. A timestamped run directory is created inside it.",
    )
    parser.add_argument("--artifact-run-name", help="Optional run directory name under --artifact-dir.")
    parser.add_argument(
        "--keep-proxy-env",
        action="store_true",
        help="Do not clear HTTP_PROXY/HTTPS_PROXY/ALL_PROXY for local base URLs.",
    )
    return parser.parse_args()


def make_local_chat_model(args: argparse.Namespace, *, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=max_tokens,
        http_client=httpx.Client(trust_env=False, timeout=args.timeout),
    )


def configure_logging(args: argparse.Namespace) -> None:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO if args.log_file else getattr(logging, args.log_level))
    handlers: list[logging.Handler] = [stream_handler]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, args.log_level))
        handlers.append(file_handler)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy_logger in ["httpx", "httpcore", "openai"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def configure_local_no_proxy(base_url: str, *, keep_proxy_env: bool = False) -> None:
    """Force localhost calls to bypass machine-wide proxy settings."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    no_proxy_hosts = set(LOCAL_HOSTS)
    if host:
        no_proxy_hosts.add(host)
    merged_no_proxy = merge_no_proxy_values(no_proxy_hosts)
    for name in NO_PROXY_ENV_VARS:
        os.environ[name] = merged_no_proxy

    if keep_proxy_env or host not in LOCAL_HOSTS:
        logging.getLogger(__name__).info("no-proxy configured: %s=%s", NO_PROXY_ENV_VARS[0], merged_no_proxy)
        return
    cleared = []
    for name in PROXY_ENV_VARS:
        if name in os.environ:
            os.environ.pop(name, None)
            cleared.append(name)
    logging.getLogger(__name__).info(
        "local no-proxy configured: %s=%s cleared_proxy_env=%s",
        NO_PROXY_ENV_VARS[0],
        merged_no_proxy,
        cleared or [],
    )


def merge_no_proxy_values(required_hosts: set[str]) -> str:
    existing: list[str] = []
    for name in NO_PROXY_ENV_VARS:
        existing.extend(part.strip() for part in os.environ.get(name, "").split(",") if part.strip())
    values = []
    seen = set()
    for value in [*existing, *sorted(required_hosts)]:
        if value not in seen:
            seen.add(value)
            values.append(value)
    return ",".join(values)


def write_summary(args: argparse.Namespace, result: object) -> None:
    if not args.summary_file:
        return
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    gepa_best_idx = getattr(result, "best_idx", None)
    best_idx = select_deployment_candidate_index(result)
    candidates = list(getattr(result, "candidates", []) or [])
    val_scores = list(getattr(result, "val_aggregate_scores", []) or [])
    tied_best_indices: list[int] = []
    if val_scores:
        max_score = max(float(score) for score in val_scores)
        tied_best_indices = [index for index, score in enumerate(val_scores) if abs(float(score) - max_score) <= 1e-12]
    best_candidate = (
        candidates[best_idx]
        if best_idx is not None and 0 <= best_idx < len(candidates)
        else getattr(result, "best_candidate", {})
    )
    final_test = None
    artifact_summary: dict[str, object] = {}
    latest_run = Path(args.artifact_dir) / "latest_run.txt" if args.artifact_dir else None
    if latest_run is not None and latest_run.exists():
        artifact_run_dir = Path(latest_run.read_text(encoding="utf-8").strip())
        final_test_path = artifact_run_dir / "final_test" / "summary.json"
        if final_test_path.exists():
            final_test = json.loads(final_test_path.read_text(encoding="utf-8"))
        artifact_summary_path = artifact_run_dir / "result_summary.json"
        if artifact_summary_path.exists():
            artifact_summary = json.loads(artifact_summary_path.read_text(encoding="utf-8"))
    summary = {
        "result_type": type(result).__name__,
        "best_idx": best_idx,
        "gepa_best_idx": gepa_best_idx,
        "tie_break_applied": best_idx != gepa_best_idx,
        "selection_policy": "incumbent_on_validation_tie",
        "tied_best_indices": tied_best_indices,
        "best_val_score": (
            result.val_aggregate_scores[best_idx]
            if best_idx is not None and hasattr(result, "val_aggregate_scores")
            else None
        ),
        "val_aggregate_scores": val_scores,
        "parents": getattr(result, "parents", None),
        "discovery_eval_counts": getattr(result, "discovery_eval_counts", None),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "overall_metric_calls": artifact_summary.get(
            "overall_metric_calls", getattr(result, "total_metric_calls", None)
        ),
        "preflight_actionability": artifact_summary.get("preflight_actionability"),
        "num_full_val_evals": getattr(result, "num_full_val_evals", None),
        "num_candidates": getattr(result, "num_candidates", None),
        "component_lengths": (
            {name: len(text) for name, text in best_candidate.items()} if isinstance(best_candidate, dict) else None
        ),
        "final_test": final_test,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
    os.environ["GEPA_CONTEXT_WINDOW_TOKENS"] = str(args.context_window_tokens)
    os.environ["GEPA_TRACE_CONTEXT_RATIO"] = str(args.trace_context_ratio)
    os.environ["GEPA_TRACE_KEEP_RATIO"] = str(args.trace_keep_ratio)
    configure_logging(args)
    configure_local_no_proxy(args.base_url, keep_proxy_env=args.keep_proxy_env)

    task_llm = make_local_chat_model(args, max_tokens=args.task_max_tokens)
    reflection_llm = make_local_chat_model(args, max_tokens=args.reflection_max_tokens)

    result = run_configured_skill_optimization(
        args.config,
        task_llm=task_llm,
        reflection_llm=reflection_llm,
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
    write_summary(args, result)

    print(f"Result type: {type(result).__name__}")
    deployment_best_idx = select_deployment_candidate_index(result)
    if hasattr(result, "val_aggregate_scores") and deployment_best_idx is not None:
        print(f"Deployment candidate: {deployment_best_idx}")
        print(f"Best val score: {result.val_aggregate_scores[deployment_best_idx]}")
    if hasattr(result, "total_metric_calls"):
        print(f"Total metric calls: {result.total_metric_calls}")
    if args.artifact_dir:
        latest_run = Path(args.artifact_dir) / "latest_run.txt"
        if latest_run.exists():
            run_dir = Path(latest_run.read_text(encoding="utf-8").strip())
            print(f"Artifacts: {run_dir}")
            artifact_summary_path = run_dir / "result_summary.json"
            if artifact_summary_path.exists():
                artifact_summary = json.loads(artifact_summary_path.read_text(encoding="utf-8"))
                if artifact_summary.get("overall_metric_calls") != artifact_summary.get("total_metric_calls"):
                    print(f"Total metric calls including preflight: {artifact_summary['overall_metric_calls']}")
            final_test_path = run_dir / "final_test" / "summary.json"
            if final_test_path.exists():
                final_test = json.loads(final_test_path.read_text(encoding="utf-8"))
                print(
                    "Final test: "
                    f"seed={final_test['seed_mean']:.3f} "
                    f"best={final_test['best_mean']:.3f} "
                    f"improvement={final_test['improvement']:.3f}"
                )
    candidates = list(getattr(result, "candidates", []) or [])
    deployment_candidate = (
        candidates[deployment_best_idx]
        if deployment_best_idx is not None and 0 <= deployment_best_idx < len(candidates)
        else getattr(result, "best_candidate", {})
    )
    if isinstance(deployment_candidate, dict):
        print("Best candidate components:")
        for name, text in deployment_candidate.items():
            print(f"- {name}: {len(text)} chars")


if __name__ == "__main__":
    main()
