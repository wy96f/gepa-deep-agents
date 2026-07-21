# 信贷审批 GEPA 示例项目

本示例用于优化中国规上企业信贷审批智能体的“项目风险点”分析能力。

每条数据只把企业名称作为 `input` 交给智能体。审批官风险评价意见中的“项目风险点”章节保存在 `data`，仅供评估器和反思步骤使用，不会在智能体运行时泄露。GEPA 根据智能体轨迹、输出与审批官意见之间的差距，优化 `skills/credit-risk-review/SKILL.md` 和 `reference/*.md` 中可复用的方法。

`SKILL.md` 保存稳定的风险识别流程。限定行业、商业模式或交易结构的经验应进入最具体的 reference；没有合适分类时进入 `reference/learned_expert_patterns.md`。经验模式用于帮助模型从已取得的信息识别风险逻辑，不是要求对每家企业机械执行的完整清单。行业只是机制示例，不是硬编码白名单。

评估器重点判断：

- 轨迹是否显示智能体成功取得了足够的相关企业证据；
- 最终输出是否覆盖审批官指出的核心风险；
- 输出是否说明事实依据、比较过程和风险传导；
- 缺少工具能力时，是否准确识别为 `TOOL_CAPABILITY_GAP`。

信息获取只认可工具调用与成功结果成对出现。提示词、技能、智能体文字或最终答案中出现关键词，都不能证明已取得证据。工具能力判断使用种子版本的工具描述，避免通过修改描述虚构工具实现并不存在的数据源。

专家 checkpoint 未覆盖时仍会降低分数。每个 checkpoint 通过 `evidence_expectations` 关联自己的运行时证据，避免一次财务查询错误地替客户、抵押等风险点解锁文本优化。框架再根据轨迹区分后续动作：没有对应工具时新增工具或 MCP；有工具但未调用时优化调用触发；参数错误时优化调用语义；运行时失败时修工具或环境；已有对应证据但未形成风险点时再优化 skill/reference。评分不会因为缺工具而放宽，文本优化也不会替工具故障背锅。

优化目标是萃取审批经验：

- 从有限但真实的企业信息中识别对应风险点；
- 说明风险点的事实依据、传导逻辑和可能影响；
- 缺少企业事实时不推测、不用通用风险清单补齐；
- 输出风险点而不输出审批意见、授信结论或放款条件；
- 保持稳定流程与有适用范围的专业经验相分离；
- 避免针对单一企业或单一行业过拟合。

示例使用混合工具能力：`lookup_financial_snapshot` 为部分企业返回独立的模拟财务数据，用于验证“已有证据但调用或分析不足”时的文本优化；客户交易、司法、抵质押等数据仍故意不提供，用于验证 `TOOL_CAPABILITY_GAP` 和工具建设清单。政策查询与风险记录工具不会被误认为企业数据源。GEPA 可以优化工具描述，但不会修改工具实现。

统一评价规则保存在 `deepagents_gepa_configs/credit_approval.toml` 的 `[dataset].rubric`，不在每条 JSONL 中重复。可使用下列命令将多份审批意见清洗为本示例格式；清洗器会读取同一份配置中的真实工具名称和 description，LLM 结构化提取可通过 `--extraction-model` 按需启用。

```bash
uv run --no-sync python examples/langchain_adapter/clean_credit_risk_dataset.py \
  --config examples/langchain_adapter/deepagents_gepa_configs/credit_approval.toml \
  --input-dir /path/to/risk-opinions \
  --output examples/langchain_adapter/deepagents_gepa_credit_approval_project/evals/project_risk_sections.jsonl
```

配置会按难度进行确定性的分层训练集、验证集和测试集划分。优化结束后，框架自动在留出的测试集上比较种子与最佳候选，并将结果保存在运行目录的 `final_test/` 中。
