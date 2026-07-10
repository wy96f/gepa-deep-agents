# Deep Agents GEPA Demo Project

This is one shared project used by both config examples:

- `manual.toml` manually lists the memory, skills, subagents, tools, MCP descriptions, and dataset to optimize.
- `langgraph_cli.toml` points at this project's `langgraph.json`; GEPA imports the configured graph entry and captures its `create_deep_agent(...)` call.

You should not copy the agent code into each TOML. Use `manual.toml` when you
want direct control over each file surface. Use `langgraph_cli.toml` when your
LangGraph CLI graph config is already the source of truth.

Runtime entrypoints in `deep_agent_skill_directory.py`:

- `load_deepagents_gepa_config(...)`: reads TOML.
- `build_candidate_from_deep_agent_project(...)`: loads project text into a GEPA candidate.
- `apply_candidate_to_deep_agent_project(...)`: writes a candidate into a temporary project tree.
- `configured_rollout(...)`: runs the materialized candidate.

The tests call these functions for both TOMLs in
`tests/test_deep_agent_skill_directory_example.py::test_repository_toml_examples_load_and_run`.

Project assets:

- `AGENTS.md`: long-term memory.
- `skills/`: main-agent skills.
- `subagents/risk-reviewer/skills/`: subagent-specific skills.
- `langgraph.json`: graph config used by `langgraph_cli.toml`.
- `langgraph_agent.py`: graph entry returning a `CompiledStateGraph` built by `create_deep_agent(...)`.
- `tools.py`: Python tools reused by both modes.
- `mcp/`: MCP server assets; GEPA optimizes declared MCP tool descriptions, not server code.
- `evals/golden.jsonl`: tiny golden dataset for smoke tests.
