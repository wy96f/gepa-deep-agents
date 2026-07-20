# Deep Agents GEPA Text-Surface Optimization

This example turns a Deep Agents project into a GEPA optimization target.
It is designed for projects where the useful behavior lives in text surfaces:

- `AGENTS.md`
- the main `system_prompt`
- tool descriptions
- subagent descriptions and prompts
- `SKILL.md`
- `reference/*.md`
- MCP tool descriptions

It does not optimize Python tool code, middleware code, arbitrary source files,
or `scripts/*.py`.

GEPA still sees the standard `dict[str, str]` candidate shape. The example code
is responsible for discovering Deep Agents text surfaces, materializing a
candidate into a temporary project directory, running Deep Agents natively, and
building Darwin/SkillOpt-style feedback.

## Files

The main implementation is:

```text
examples/langchain_adapter/deep_agent_skill_directory.py
```

The local llama.cpp/OpenAI-compatible runner is:

```text
examples/langchain_adapter/run_deepagents_gepa_local.py
```

The framework support package is:

```text
examples/langchain_adapter/deepagents_gepa/
```

It contains the small Deep Agents-specific abstractions used by this example:

- `framework.py`: protocols plus default implementations for dataset providers,
  evaluators, selectors, reflection templates, constraints, and candidate
  materializers. There is no separate `DeepAgentsRunner` abstraction; this
  framework always runs through Deep Agents.
- `artifacts.py`: run artifact persistence for configs, datasets, agent
  rollouts, proposal prompts, rejected proposals, candidates, summaries, and
  materialized best candidates.

The default implementations are used by `run_configured_skill_optimization(...)`:

- `DefaultDatasetProvider`
- `DefaultEvaluator`
- `DefaultFeedbackComponentSelector`
- `DefaultReflectionTemplateRegistry`
- `DefaultConstraintSet`
- `DefaultCandidateMaterializer`

The runnable demo project is:

```text
examples/langchain_adapter/deepagents_gepa_demo_project/
```

Its golden dataset intentionally includes multiple examples for each support
route (`billing`, `account`, `engineering`, `product`) so small train/validation
splits do not leave an entire class represented by only one held-out example.

另有一个信贷审批专家风险意见萃取示例：

```text
examples/langchain_adapter/deepagents_gepa_credit_approval_project/
```

Runnable configs are:

```text
examples/langchain_adapter/deepagents_gepa_configs/manual.toml
examples/langchain_adapter/deepagents_gepa_configs/langgraph_cli.toml
examples/langchain_adapter/deepagents_gepa_configs/credit_approval.toml
```

The generic template is:

```text
examples/langchain_adapter/deepagents_gepa_example.toml
```

`deepagents_gepa_example.toml` is a copy-and-edit template for a real project.
It is not the primary demo config used by tests. Use `manual.toml` or
`langgraph_cli.toml` when you want to run the repository demo.

## 信贷审批示例

信贷审批示例用于萃取专家经验，不要求匹配唯一标准答案。每条数据只把企业名称作为
`input` 交给智能体。审批官风险评价意见中的“项目风险点”章节保存在 `data`，仅供
评估器和反思步骤使用，不会在智能体运行时泄露。评审模型比较智能体轨迹、输出与
专家风险意见，判断是否取得了正确的企业证据、覆盖核心风险点并讲清风险逻辑。
技能目录包含以下可复用方法论：

```text
skills/credit-risk-review/SKILL.md
skills/credit-risk-review/reference/financial_statement_analysis.md
skills/credit-risk-review/reference/cashflow_and_repayment.md
skills/credit-risk-review/reference/collateral_and_guarantee.md
skills/credit-risk-review/reference/industry_management_and_warnings.md
skills/credit-risk-review/reference/learned_expert_patterns.md
```

`SKILL.md` 负责稳定的审查流程：形成假设、取得证据、判定状态、分析风险传导并形成
审批措施。有行业、商业模式或交易结构适用范围的经验应进入最具体的
`reference/*.md`。`learned_expert_patterns.md` 是预先声明的专家经验沉淀面，用于
容纳无法归入现有分类的可复用模式。一次 GEPA 优化中的 component key 集合固定，
不能在中途凭空新增文件；领域需要独立知识分类时，应在运行前预建相应 reference。

评审与反思模板要求经验模式写清机制假设、适用信号、排除条件、动态取证计划、分析
方法、状态判定、风险传导和审批应用。这些不是所有企业共用的一张固定清单：智能体应
根据当前企业的商业模式、交易、资产和融资特征选择证据及比较基准。行业案例只用于
解释经济机制，不是封闭名单。这样可以避免某条规则改善一个样本后被无条件应用到无关
企业。

数据格式如下：

```json
{
  "input": "华东钢铁集团有限公司",
  "data": "七、项目风险点\n1、钢铁行业周期性风险...",
  "rubric": "评价智能体是否自主取得相关证据、覆盖专家风险点并讲清风险逻辑。",
  "metadata": {
    "checkpoints": [
      {"label": "钢铁行业周期性风险", "keywords": ["钢铁行业", "周期", "库存减值"]},
      {"label": "高负债规模与债务结构压力风险", "keywords": ["资产负债率", "短贷长投"]}
    ],
    "trace_expectations": [
      {
        "label": "行业周期信息获取",
        "tool_names": ["lookup_industry_cycle"],
        "tool_intent_keywords": ["行业", "钢铁", "周期"]
      },
      {"label": "债务结构信息获取", "tool_intent_keywords": ["负债", "借款", "融资"]}
    ]
  }
}
```

数据没有 `expected` 标准答案。评审模型评价轨迹是否显示相关信息获取、最终输出是否
覆盖专家风险点，以及是否说明风险传导。`metadata` 可选但很有用：
`metadata.checkpoints` 是最终答案的严格覆盖清单；trace expectation 只有在轨迹中
出现成对的成功工具调用与结果，且种子工具能力与该期望匹配时才算完成。提示词、
SKILL.md、智能体文字和最终答案中的关键词都不能作为已取得证据。已知目标工具时可
设置 `tool_names`；否则框架使用种子工具名称、描述和 `tool_intent_keywords` 进行
保守匹配。模糊匹配至少需要两个独立能力关键词，避免把内部政策查询误判为企业数据
查询。

工具能力缺口与文本缺陷可以同时存在。案例既缺少外部数据工具、又漏掉可复用的专家
分析方法时，框架仍允许优化相应 reference，同时在 feedback 和运行产物中保留
`TOOL_CAPABILITY_GAP` 供工具开发排期。只有不存在可优化文本问题的纯工具缺口才跳过
文本 mutation。

批量清洗审批意见：

```bash
uv run --no-sync python examples/langchain_adapter/clean_credit_risk_dataset.py \
  --input-dir /path/to/risk-opinions \
  --output examples/langchain_adapter/deepagents_gepa_credit_approval_project/evals/project_risk_sections.jsonl
```

运行优化：

```bash
uv run --no-sync python examples/langchain_adapter/run_deepagents_gepa_local.py \
  --config examples/langchain_adapter/deepagents_gepa_configs/credit_approval.toml \
  --base-url http://127.0.0.1:8080/v1 \
  --model local-chat-model \
  --context-window-tokens 200000 \
  --trace-context-ratio 0.12 \
  --max-metric-calls 10 \
  --num-threads 1 \
  --artifact-dir examples/langchain_adapter/runs \
  --artifact-run-name credit_approval_10
```

分析运行产物：

```bash
uv run --no-sync python examples/langchain_adapter/analyze_deepagents_gepa_run.py \
  --run-dir examples/langchain_adapter/runs/credit_approval_10
```

## Loading Modes

The example now exposes two user-facing loading modes.

### Manual Mode

Manual mode is for projects where you want to declare the Deep Agents runtime
pieces directly in TOML:

```toml
[agent]
mode = "manual"
project_root = "../deepagents_gepa_demo_project"
system_prompt = "You are a support router loaded from manual config."
memory = ["AGENTS.md"]
skills = ["skills"]
tools = ["tools:tag_ticket"]

[[agent.subagents]]
name = "risk-reviewer"
description = "Use to review ambiguous routing decisions before finalizing."
system_prompt = "You review routing risk. Use lookup_policy and the risk-review skill."
tools = ["tools:lookup_policy"]
skills = ["subagents/risk-reviewer/skills"]
```

Manual mode can also declare explicit surfaces:

```toml
[surfaces.memory]
kind = "file"
path = "AGENTS.md"
component = "memory:AGENTS.md"
source_type = "memory"

[surfaces.skills]
kind = "skill_dir"
path = "skills"

[surfaces.risk_reviewer_skills]
kind = "skill_dir"
path = "subagents/risk-reviewer/skills"
owner = "risk-reviewer"
```

Use this mode when the project is not already exposed through LangGraph CLI, or
when you want very explicit control over which files, tools, skills, subagents,
and MCP tool descriptions are part of the optimization surface.

### Path Resolution

Config paths are resolved in two steps:

```text
project_root      relative to the TOML file
memory            relative to project_root
skills            relative to project_root
subagent skills   relative to project_root
surfaces.*.path   relative to project_root
dataset.path      relative to project_root
langgraph_config  relative to project_root
```

For example, `examples/langchain_adapter/deepagents_gepa_configs/manual.toml`
uses:

```toml
project_root = "../deepagents_gepa_demo_project"
memory = ["AGENTS.md"]
skills = ["skills"]
```

This resolves to:

```text
examples/langchain_adapter/deepagents_gepa_demo_project/AGENTS.md
examples/langchain_adapter/deepagents_gepa_demo_project/skills/
```

During a rollout, the candidate is materialized into a temporary project tree.
Deep Agents then receives native paths such as `memory=["AGENTS.md"]` and
`skills=["skills"]`, rooted at that temporary project.

### LangGraph CLI Mode

LangGraph CLI mode is for projects that already expose a graph through
`langgraph.json`.

The TOML stays small:

```toml
[agent]
mode = "langgraph_cli"
project_root = "../deepagents_gepa_demo_project"
langgraph_config = "langgraph.json"
graph = "support_router"
```

The demo `langgraph.json` uses the normal LangGraph CLI graph reference shape:

```json
{
  "dependencies": ["."],
  "graphs": {
    "support_router": "./langgraph_agent.py:support_router"
  }
}
```

The graph entry returns a `CompiledStateGraph` created by `create_deep_agent`.
There is no `gepa_deep_agent_spec` sidecar. During graph loading, the example
temporarily captures calls to:

- `deepagents.create_deep_agent`
- `deepagents.graph.create_deep_agent`

It then builds the GEPA candidate from the captured `create_deep_agent(...)`
arguments. This keeps the GEPA harness aligned with the actual Deep Agents graph
used by the project.

Supported graph entries follow the LangGraph CLI shape:

```text
./your_package/your_file.py:compiled_graph_variable
./your_package/your_file.py:make_graph
```

If the graph entry is callable, the loader first calls it with a
`RunnableConfig`-shaped `{}` and falls back to a no-argument call.

## Candidate Components

The candidate keys follow stable names so feedback can recommend exactly what
to mutate next:

```text
memory:AGENTS.md
main:system_prompt
main:tool:<tool_name>:description
subagent:<name>:description
subagent:<name>:system_prompt
subagent:<name>:tool:<tool_name>:description
skill:<skill_name>:SKILL.md
skill:<skill_name>:reference/<file>.md
subagent:<name>:skill:<skill_name>:SKILL.md
subagent:<name>:skill:<skill_name>:reference/<file>.md
mcp:tool:<tool_name>:description
```

The example excludes:

```text
scripts/*.py
tool function bodies
middleware implementation code
arbitrary source-code AST rewriting
reflection_prompt_template
```

`reflection_prompt_template` is optimizer scaffolding. It tells GEPA how to ask
the reflection model to rewrite a component. It is intentionally fixed in this
example. If you want to optimize reflection templates themselves, treat that as
a separate meta-optimization problem with its own eval.

The default reflection prompt follows a two-step pattern:

```text
global diagnosis -> scoped component replacement
```

The reflection model sees the candidate excerpt and feedback so it can diagnose
the project globally, but it must still return only the selected component as a
drop-in replacement. For long evaluation contexts, the template repeats the
authoritative component key and current component after the evaluation data so
the model does not accidentally return a neighboring reference or skill. Passed
constraints and unchanged size metrics are omitted from proposer input; their
complete values remain in rollout artifacts. The prompt asks for an explicit
proposal rationale before the final fenced replacement:

Company-name keywords are treated only as weak discovery clues. Learned rules
must require observable business or transaction evidence, and numeric
thresholds must come from policy/evidence or be labeled as adjustable stress
assumptions.

```text
Failure pattern
Evidence across examples
Selected component
Why this component
Why not other components
Applicability scope and exclusions
Cross-case regression risk
Operational rule shape
Boundary checks
Hidden-data boundary check
Intended behavior change
```

This is a review artifact, not hidden chain-of-thought. GEPA still extracts only
the final fenced block as the new component text. `<side_info>`, expert data,
rubrics, checkpoints, and evaluator feedback are optimizer-only evidence; they
were never runtime input. The reflection model is explicitly forbidden from
claiming the agent saw or failed to read them.

## Skill Scripts

Skill scripts are runtime resources, not candidate text.

If `SKILL.md` contains a command such as:

```text
python scripts/check_route.py
```

the example keeps `scripts/check_route.py` out of the candidate, but copies it
into the temporary skill tree so Deep Agents can still execute it. It also
creates a temporary workspace-level alias such as:

```text
scripts/check_route.py
```

This supports existing skills that assume `python scripts/foo.py` runs from the
agent workspace root.

The rollout uses Deep Agents `LocalShellBackend`:

```python
LocalShellBackend(root_dir=temp_root, virtual_mode=True, inherit_env=True)
```

This is useful for a local optimization harness because Deep Agents' `execute`
tool can run skill scripts. It is also real host shell execution. Use a more
restricted backend for untrusted skills or production environments.

When candidates are applied to a temp/output directory, managed skill source
directories are mirrored fresh from the seed project. This prevents stale files
from previous candidate rounds from lingering on disk.

## MCP

MCP servers and MCP tool descriptions can be declared in TOML:

```toml
[[mcp.servers]]
name = "routing-risk"
transport = "stdio"
command = "python mcp/routing_risk_server.py"

[[mcp.tools]]
name = "search_routing_risks"
server = "routing-risk"
description = "Search routing risk notes for missing evidence and likely misroutes."
```

MCP tool descriptions become candidate components:

```text
mcp:tool:search_routing_risks:description
```

The example does not automatically launch arbitrary MCP servers. Instead,
`run_configured_skill_optimization(...)` accepts an `mcp_loader` hook. That hook
receives the declared servers plus the current candidate's MCP tool descriptions
and returns LangChain tools to append to the Deep Agents runtime.

## Dataset Sources

Golden examples use JSONL rows:

```json
{"input": "...", "data": "...", "expected": "...", "rubric": "...", "metadata": {"topic": "..."}}
```

Fields mean:

- `input`: the user task or question.
- `data`: optional evaluator-only expert material, such as an approval
  officer's project-risk section. It is not passed to the agent during rollout.
- `expected`: optional known answer, route, label, or structured result.
- `rubric`: optional evaluation guidance for open-ended tasks.
- `metadata`: optional grouping, topic, difficulty, source, or trace metadata.
- top-level `split` or `metadata.split`: optional explicit `train`, `val`, or
  `test` assignment.
- top-level `stratum` or `metadata.stratum`: optional grouping used by
  stratified splitting.

For deterministic tasks, `expected` is useful. For open-ended work such as due
diligence report generation, `rubric` is usually more valuable than exact-match
text.

For expert-experience distillation, `data` can hold the expert section and
`metadata.checkpoints` can make open-ended examples harder and less prone to
score saturation:

```json
{
  "input": "江北化工新材料股份有限公司",
  "data": "七、项目风险点\n1、技改项目合规闭环风险...",
  "rubric": "评价 agent 是否自主获取相关信息, 覆盖专家风险点, 并讲清风险逻辑。",
  "metadata": {
    "checkpoints": [
      {"label": "环评安评提款前置", "keywords": ["环评", "安全验收", "提款前置"]},
      {"label": "客户集中压力测试", "keywords": ["三家大型客户", "客户集中", "集中度压力测试"]}
    ],
    "trace_expectations": [
      {"label": "环保安监信息获取", "tool_intent_keywords": ["环保", "安全生产", "环评"]},
      {"label": "客户交易信息获取", "tool_intent_keywords": ["客户", "订单", "回款"]}
    ]
  }
}
```

Each checkpoint is a reusable expert judgment point. The evaluator reports
matched and missing checkpoints, caps open-ended scores when checkpoints are
missing, and pushes feedback toward the most specific skill/reference component
that should preserve the lesson. Trace expectations remain diagnostics rather
than a hard score gate, but their matched/missing state is deterministic and
uses only successful tool evidence. Failed tool results are retained separately.

Dataset splitting is deterministic and stratified by default:

```toml
[dataset]
split_strategy = "stratified"
train_ratio = 0.60
val_ratio = 0.20
test_ratio = 0.20
stratify_by = ["metadata.difficulty", "metadata.industry"]
seed = 17
evaluate_final_test = true
```

Explicit split labels take precedence. Unlabeled rows are distributed by the
configured strata using a stable hash, so JSONL ordering does not place whole
industries only in train or test. After GEPA finishes, the harness evaluates
seed and the deployment candidate on the held-out test split. This final test
does not influence optimization, acceptance, or Pareto selection.

Langfuse import supports two dataset styles:

- `langfuse_experience`: imports user questions, corrections, follow-ups, and
  expert risk probes as experience. The final assistant answer is not assumed
  to be correct.
- `langfuse_labeled`: keeps traces only when they have explicit labels, scores,
  accepted outputs, or human feedback.

This lets online conversations improve skills and references even when the
production answer was imperfect. The user questions themselves often contain the
expert judgment worth preserving.

## Reusable Extension Points

Most projects should reuse Deep Agents loading, candidate discovery/application,
artifact persistence, and materialization. Override only the parts that truly
vary by domain:

- `DatasetProvider`: how examples are loaded and split. Use this for golden
  JSONL, Langfuse traces, sampled production conversations, or synthetic tasks.
- `Evaluator`: how one rollout becomes a score and feedback. Use this when the
  domain needs a custom judge rubric, such as credit approval officer comments.
- `ReflectionTemplateRegistry`: how the reflection model is instructed to edit
  each component. Use this when a domain has special component boundaries or
  writing style, while keeping the final fenced-block contract.
- `ComponentSelector`: which component should be mutated next. Use this when
  feedback should prefer a specific family, such as `reference/*.md` for expert
  methodology extraction.
- `Constraint`: generic hard/advisory checks for candidate validity. Keep hard
  checks high-confidence and domain-neutral when possible; put softer judgment
  into the evaluator.

`run_configured_skill_optimization(...)` accepts these as optional Python hooks:

```python
run_configured_skill_optimization(
    config_path,
    task_llm,
    reflection_llm,
    dataset_provider=...,
    evaluator=...,
    template_registry=...,
    component_selector=...,
    constraint_policy=...,
)
```

Everything else can usually remain the default implementation.

## Evaluation And Feedback

The evaluator borrows useful ideas from `darwin-skill`, `hermes-agent-self-evolution`,
and SkillOpt, but keeps GEPA's standard full-text candidate and Pareto search.

The configured runner uses the reflection model as the primary judge by
default. One judge call returns the score, failure classification, recommended
component, boundary assessment, and concise feedback. Deterministic rules are
kept deliberately narrow and mostly act as safety caps.

Hard deterministic constraints should be generic and high-confidence:

```text
component is non-empty
component is below a broad size limit
SKILL.md has required YAML frontmatter with name/description
referenced skill scripts exist in the materialized workspace
explicit foreign-runtime text is absent
non-skill prompt/description did not paste SKILL.md frontmatter
component did not paste candidate-excerpt labels such as "### skill:..."
component did not paste bare candidate keys such as "main:system_prompt"
```

Advisory checks are recorded for the judge but do not hard-fail a candidate:

```text
growth limit
tool description detail
skill workflow/failure-mode/guardrail quality
most component-boundary and style judgments
```

The judge evaluates:

- hard correctness signals
- soft rubric/task completion signals
- mixed/composite signals
- with-candidate vs baseline behavior
- structure quality
- actionable specificity
- runtime neutrality
- size and growth gates
- hard constraints and advisory notes
- skill/reference path validity
- tool/subagent description non-emptiness

If an example has an `expected` answer but the agent does not produce the
required structured answer, the deterministic fallback caps the composite score.
The reflection judge is also capped by that correctness rule, so a fluent answer
that does not return the expected route/label cannot be accepted as a high-score
improvement. For expert-data rows with `metadata.checkpoints`, the judge
is also capped by checkpoint coverage; missing expert points are classified as
`SKILL_DEFECT` so the next proposal is encouraged to update `SKILL.md` or a
focused `reference/*.md` file. If the missing point requires evidence that no
declared seed tool can obtain, it is classified as `TOOL_CAPABILITY_GAP`
instead. If a hard deterministic gate fails, the final
judge score is capped to zero. Advisory notes do not cap the score by
themselves; they are fed to the judge so it can decide whether the issue
actually matters.

Hard boundary gates catch only high-confidence bad proposals, including:

```text
system_prompt contains SKILL.md YAML frontmatter
component includes "### skill:..." candidate-excerpt labels
component starts with or contains a bare candidate key line such as "main:system_prompt"
```

Other boundary questions, such as whether a prompt is too verbose or whether a
skill copied too much reference knowledge, are judge feedback rather than hard
rules to avoid false positives.

Feedback includes:

- per-dimension scores
- raw composite score and score caps
- gate failures
- with-candidate output
- baseline output
- adaptive trace summary
- trace expectation and tool capability diagnostics
- weakest dimension
- failure classification
- recommended component key
- short reason for the recommendation
- knowledge scope and applicability conditions
- cross-case regression risk
- an operational rule shape: trigger -> evidence -> analysis -> transmission -> action

Failures are classified as:

```text
SKILL_DEFECT
EXECUTION_LAPSE
TOOL_CAPABILITY_GAP
NO_FAILURE
```

`SKILL_DEFECT` means the available skill/reference/tool text is missing,
ambiguous, or wrong. It tends to recommend:

```text
skill:...:SKILL.md
skill:...:reference/...
subagent:...:skill:...:SKILL.md
...:tool:...:description
```

`EXECUTION_LAPSE` means the needed guidance exists but the agent did not use it
reliably. It tends to recommend:

```text
memory:AGENTS.md
main:system_prompt
subagent:<name>:system_prompt
subagent:<name>:description
```

`TOOL_CAPABILITY_GAP` means the required external evidence has no matching
current tool capability. It recommends no text component. Capability-gap-only
minibatches return an empty component selection, so the unchanged proposal is
rejected rather than teaching prompts to invent unavailable data. The gap is
still saved for the tool/MCP backlog.

The component selector aggregates recommendations across feedback records. It
prefers component keys that appear most often in low-scoring trajectories. If
the same component is repeatedly selected for the same candidate without
producing an accepted improvement, it cools that component down and tries
another surface. If no valid key is found, it falls back to round-robin
selection. `TOOL_CAPABILITY_GAP` trajectories are excluded from this vote.

When the selected component is an explicitly managed learned/expert/experience
reference, the selector first checks whether its owning `SKILL.md` already
routes to that reference. If it does, only the reference is mutated. Otherwise,
the selector also mutates the owning `SKILL.md` so the workflow can add the
missing applicability trigger and lookup step. Ordinary reference files remain
single-component mutations. This avoids both unread learned knowledge and the
failure mode where a model copies the reference body into an already-correct
`SKILL.md`.

The default reflection minibatch size is `3`, so a proposal normally sees more
than one trajectory and must explain evidence across examples. For small,
heterogeneous datasets, keep validation coverage across multiple industries;
scope a rule by observable borrower signals when it helps one segment but could
reduce quality in another.

GEPA's own acceptance and Pareto frontier act as the in-memory ratchet. This
example does not implement SkillOpt's patch schema, hierarchical merge,
rank-and-select, learning-rate scheduler, sleep cycle, transcript mining, or
experience replay.

## Run With Local llama.cpp

Start llama.cpp with an OpenAI-compatible server and a large enough context
window. Deep Agents adds substantial system context, so 2048 tokens is too
small even for the minimal graph. The local test run used:

```text
n_ctx = 131072
```

Run manual mode:

```bash
cd /Users/yangwei/pycharm/gepa-deep-agents
source .venv/bin/activate
python examples/langchain_adapter/run_deepagents_gepa_local.py \
  --config examples/langchain_adapter/deepagents_gepa_configs/manual.toml \
  --base-url http://127.0.0.1:8080/v1 \
  --model local-chat-model \
  --context-window-tokens 200000 \
  --trace-context-ratio 0.12 \
  --max-metric-calls 10 \
  --num-threads 1 \
  --artifact-dir examples/langchain_adapter/runs
```

Run LangGraph CLI auto-discovery mode:

```bash
python examples/langchain_adapter/run_deepagents_gepa_local.py \
  --config examples/langchain_adapter/deepagents_gepa_configs/langgraph_cli.toml \
  --base-url http://127.0.0.1:8080/v1 \
  --model local-chat-model \
  --context-window-tokens 200000 \
  --trace-context-ratio 0.12 \
  --max-metric-calls 10 \
  --num-threads 1 \
  --artifact-dir examples/langchain_adapter/runs
```

The runner creates `ChatOpenAI` clients with:

```python
httpx.Client(trust_env=False)
```

This avoids local `127.0.0.1` calls accidentally going through an HTTP proxy.
For local base URLs, the runner also forces `NO_PROXY/no_proxy` to include
`127.0.0.1`, `localhost`, and `::1`, and clears proxy environment variables in
the current Python process unless `--keep-proxy-env` is set.

The local runner defaults to large completion budgets:

```text
--task-max-tokens 65536
--reflection-max-tokens 131072
```

Trace handling is adaptive and follows the same staged idea as Deep Agents'
summarization middleware, without rewriting the live agent state. Full raw
rollout messages are always saved under `agent_logs/rollouts/*.json`. The
judge/reflection copy first removes low-value `write_file` and `edit_file`
calls, arguments, and tool results. It retains AI message text and useful tool
calls with their query arguments. Only after that final evaluation trace
exceeds its budget does the reflection model summarize the older messages; a
recent whole-message tail is appended unchanged. There is no character slicing
of individual AI messages or tool results. If the summarizer is unavailable,
the framework keeps the complete filtered trace instead of silently truncating
it. By default the trace can use about 12% of a 200k-token context window and
keeps the most recent 10% of that trace budget verbatim:

```text
--context-window-tokens 200000
--trace-context-ratio 0.12
--trace-keep-ratio 0.10
```

The same values can be set through environment variables:

```text
GEPA_CONTEXT_WINDOW_TOKENS=200000
GEPA_TRACE_CONTEXT_RATIO=0.12
GEPA_TRACE_KEEP_RATIO=0.10
```

The default noise-tool list can be overridden when a project treats file
mutation as evaluation evidence:

```text
GEPA_TRACE_OMIT_TOOL_NAMES=edit_file,write_file
```

It also uses the reflection model as the evaluation judge by default. Disable
that and use the deterministic fallback evaluator with:

```text
--no-reflection-judge
```

If a run still fails with `ConnectError: [Errno 1] Operation not permitted`
after no-proxy is configured, the process itself is not allowed to open the
local TCP socket. Run the same command from PyCharm or a normal terminal rather
than a restricted execution sandbox.

## Run Artifacts

Runs can persist artifacts through `--artifact-dir` or the
`artifact_dir=` argument to `run_configured_skill_optimization(...)`.

The local runner defaults to:

```text
examples/langchain_adapter/runs/<timestamped-run>/
```

Each run directory contains:

```text
config/
  <original-config>.toml
  resolved_config.json
datasets/
  train.jsonl
  val.jsonl
  test.jsonl
project/
  surface_manifest.json
  seed_candidate_keys.json
candidates/
  0000/
    candidate.json
    manifest.json
    metadata.json
    components/*.txt
    diff_against_seed.patch
    diff_against_parent.patch
    diffs/*.patch
best_candidate/
  candidate.json
  manifest.json
  components/*.txt
  diff_against_seed.patch
  diff_against_parent.patch
rejected_candidates/
  <candidate-index>/
    candidate.json
    metadata.json
    diff_against_seed.patch
    diff_against_parent.patch
agent_logs/
  rollouts.jsonl
  rollouts/*.json
proposals/
  index.jsonl
  <iteration>/
    candidate.json
    manifest.json
    metadata.json
    reflective_dataset.json
    new_instructions.json
    proposal_rationale.json
    proposal_rationale_missing.json
    proposal_rationale/*.txt
    diff_against_seed.patch
    diff_against_parent.patch
    diffs/*.patch
    prompts/*.txt
    raw_lm_outputs/*.txt
rejected_proposals/
  index.jsonl
  <iteration>/
    candidate.json
    manifest.json
    metadata.json
    proposal_rationale.json
    proposal_rationale_missing.json
    diff_against_seed.patch
    diff_against_parent.patch
reflection_errors/
  index.jsonl
  <call-index>.json
  <call-index>.prompt.txt
final_test/
  seed.json
  best.json
  summary.json
materialized_best_candidate/
  AGENTS.md
  skills/
  subagents/
result_summary.json
```

`materialized_best_candidate/` is a temporary-project-style export of the best
candidate. It is meant for review and diffing. The framework does not write the
best candidate back into the source project automatically.

GEPA's `best_idx` chooses the first candidate with the maximum validation
score. The deployment harness now follows the same conservative rule: a newer
accepted candidate that merely ties does not replace the incumbent. All
accepted candidates and diffs remain available for review.
`result_summary.json` records the deployment choice as `best_idx`, preserves
GEPA's original choice as `gepa_best_idx`, and includes `selection_policy`,
`tie_break_applied`, and `tied_best_indices`. The held-out test set remains
diagnostic and is never used for candidate selection.

`agent_logs/` records each rollout: input, expected answer or rubric, final
agent response, baseline response, score, fitness dimensions, constraints, and a
serializable raw message trace. Evaluator mutations such as `fitness` are
written back to the original rollout state before artifact export. The log also
records the available and seed-capability tool inventories, successful/failed
tool evidence, matched/missing trace expectations, and tool capability gaps. The
feedback prompt uses the filtered, adaptive evaluation trace, while the raw
trace remains in the detailed rollout artifact for audit. Saving that raw file
is not part of runtime summarization and is not required by the reflection
model; it exists only when artifacts are enabled and is intended for human or
offline analysis.

Candidates that cannot load at runtime, including `SKILL.md` files with missing
or invalid YAML frontmatter or missing `name`/`description`, receive a zero
constraint cap. The harness skips Deep Agent creation for those candidates and
records `candidate_runtime_skipped` with the failed gate, avoiding repeated
Deep Agents loader warnings and wasted model calls.

The proposer receives a compact reflective record: agent output, baseline,
adaptive trace, and structured feedback are each included once. Duplicate
copies embedded inside evaluator feedback and large tool-evidence objects are
removed only from the proposer input; the complete rollout artifact is
unchanged. Provider failures during reflection are written to
`reflection_errors/` with the component, exception, prompt size, and full
prompt, so a proposal that stops at `started` can be diagnosed without the
external process log.

`proposals/` records every reflective proposal, including the rendered
reflection prompt, raw LLM output, explicit proposal rationale, and diffs
against both the parent candidate and the seed candidate.
If the reflection model starts directly with the final fenced block and omits
the review rationale, the proposal is marked with
`proposal_rationale_missing.json` and `missing_proposal_rationale` metadata.
`rejected_proposals/` is the important negative-evidence set: proposals rejected
by GEPA's subsample acceptance check are saved even though they never enter the
final candidate pool.

Rejected proposal summaries are also injected into later reflection prompts as
short negative evidence. The prompt tells the model not to copy rejected text,
only to avoid repeating the same failure pattern.

After a run, summarize effectiveness and failure patterns with:

```bash
python examples/langchain_adapter/analyze_deepagents_gepa_run.py
```

By default it reads `examples/langchain_adapter/runs/latest_run.txt`. To analyze
a specific run:

```bash
python examples/langchain_adapter/analyze_deepagents_gepa_run.py \
  --run-dir examples/langchain_adapter/runs/manual_10_agent_logs_rejected_prompt
```

The analyzer reports baseline score, best score, improvement, proposal status
counts, rejected proposal patterns, missing proposal-rationale markers, runtime
errors, missing trace expectations, tool capability gaps, and whether the run is
valid for algorithm-effectiveness analysis. If every rollout failed with a
local-model connection error, it says so explicitly instead of treating the
scores as useful.

`proposals/index.jsonl` is intentionally a lifecycle event stream and may hold
started, proposed, evaluated, and terminal rows for one iteration. The analyzer
reports both raw event count and deduplicated proposal count, using only the
latest row per iteration for status/component statistics. Final-test scores are
reported separately from optimization rollouts.

Tool capability gaps mean the evaluator expected a data-acquisition direction
but the original tool names/descriptions did not cover it. Capability checks
use seed descriptions, not optimized descriptions, because rewriting text
cannot add a data source to unchanged tool code. Those gaps are outside GEPA's
text-only optimization surface: use them as a backlog for new tools or MCP
integrations. "Missed supported expectations" are different: the tool seems
available, but no matching successful result appeared in the trace, so
skill/prompt/tool-description optimization can plausibly help.

The base artifact directory also gets:

```text
latest_run.txt
```

which points at the most recent timestamped run directory.

## Run Or Debug In PyCharm

Use this Run/Debug configuration:

```text
Script path:
/Users/yangwei/pycharm/gepa-deep-agents/examples/langchain_adapter/run_deepagents_gepa_local.py

Working directory:
/Users/yangwei/pycharm/gepa-deep-agents

Python interpreter:
/Users/yangwei/pycharm/gepa-deep-agents/.venv/bin/python
```

Manual mode parameters:

```text
--config examples/langchain_adapter/deepagents_gepa_configs/manual.toml
--base-url http://127.0.0.1:8080/v1
--model local-chat-model
--context-window-tokens 200000
--trace-context-ratio 0.12
--max-metric-calls 10
--num-threads 1
--artifact-dir examples/langchain_adapter/runs
```

LangGraph CLI mode parameters:

```text
--config examples/langchain_adapter/deepagents_gepa_configs/langgraph_cli.toml
--base-url http://127.0.0.1:8080/v1
--model local-chat-model
--context-window-tokens 200000
--trace-context-ratio 0.12
--max-metric-calls 10
--num-threads 1
--artifact-dir examples/langchain_adapter/runs
```

Useful breakpoints:

```text
run_configured_skill_optimization(...)
build_candidate_from_deep_agent_project(...)
build_deep_agent_spec_from_langgraph_config(...)
configured_rollout(...)
evaluate_response(...)
reflective_record(...)
```

## Run With The Generic CLI

The original example file also has a generic CLI:

```bash
uv run python examples/langchain_adapter/deep_agent_skill_directory.py \
  --config examples/langchain_adapter/deepagents_gepa_configs/manual.toml
```

That path uses `langchain.chat_models.init_chat_model(...)` from JSON CLI
arguments. It is fine for hosted models or simple model kwargs. For local
OpenAI-compatible endpoints, prefer `run_deepagents_gepa_local.py` because it
can pass a real `httpx.Client(trust_env=False)` object.

## Test Coverage

The main test file is:

```text
tests/test_deep_agent_skill_directory_example.py
```

It covers:

- manual config loading
- LangGraph CLI config loading
- credit approval expert-risk-section demo loading
- Python hook overrides for dataset, evaluator, templates, selector, and constraints
- auto-discovery from a `create_deep_agent(...)` graph
- candidate discovery for `AGENTS.md`
- main and subagent tool descriptions
- main and subagent skills
- `SKILL.md`
- `reference/*.md`
- MCP tool descriptions
- exclusion of `scripts/*.py`
- candidate materialization into a temp Deep Agents workspace
- preservation of subagent-specific skill sources
- copying scripts as runtime files rather than candidate text
- failure classification into `SKILL_DEFECT`, `EXECUTION_LAPSE`, and
  `TOOL_CAPABILITY_GAP`
- successful-tool-only trace acquisition matching
- evaluator fitness write-back to rollout artifacts
- lifecycle-event deduplication in the run analyzer
- deterministic stratified train/validation/test splitting
- automatic held-out seed/best final-test evaluation
- suggested component aggregation
- repeated-component cooldown
- correctness score caps
- reflection-judge correctness caps
- rubric checkpoint coverage caps
- component-boundary hard gates
- bare candidate-key boundary gates
- advisory constraints that do not hard-fail candidates
- reflection judge JSON fallback behavior
- memory reflection template anti-copy guidance
- proposal rationale, missing-rationale markers, and seed/parent diff artifacts
- artifact export of configs, datasets, candidates, and materialized best files
- dry-run fallback behavior

Run:

```bash
uv run --frozen pytest tests/test_deep_agent_skill_directory_example.py -q
```

Expected result:

```text
52 passed
```

## Production Notes

This example keeps GEPA's full-text replacement model because it is simple and
matches the existing candidate API. If you plan to optimize real long-term
`AGENTS.md` or hand-maintained `SKILL.md` files, consider a protected managed
block pattern:

```text
<!-- GEPA_LEARNED_BLOCK_START -->
...
<!-- GEPA_LEARNED_BLOCK_END -->
```

That preserves user-authored text outside the managed block while still letting
GEPA evolve learned instructions.

For production persistence, review candidate changes through your normal git or
release workflow. GEPA's Pareto state is an optimizer ratchet, not a deployment
approval system.
