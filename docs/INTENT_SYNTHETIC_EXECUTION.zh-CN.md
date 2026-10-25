# 合成本地只读意图执行（默认关闭）

本阶段提供独立宿主的最小 API 闭环：提案 → 独立确认 → 单独执行请求 → 原子准入现有 RunEngine → 查询。只运行内置整数翻倍 fixture，使用临时数据的演示不打开 socket、不调用模型或外部服务。默认 RAG API/UI 仍只生成提案和记录确认。

## 显式启用

可信宿主先调用 `register_workflows(registry)`，再调用 `register_synthetic_intent_agent(registry)`，取得绑定当前注册表及发布版本的 `SyntheticIntentBinding`。传给独立宿主：

```python
app = create_agent_app(
    runtime_factory,
    authenticate=authenticate,
    refresh_identity=refresh_identity,
    intent_interpreter=OfflineStructuredIntent(),
    synthetic_intent=binding,
)
```

`runtime_factory` 必须使用同一注册表。省略 `synthetic_intent` 时不安装执行表，也不挂载执行接口；不能通过模型输出、业务 JSON、环境变量或 scope 字段启用。宿主仍负责认证与 Agent ACL；绑定对象不是凭据。

只允许固定的 `intent-synthetic` Agent、内置算术 handler 与 fixed 工作流。校验实际实现引用、完整声明和发布摘要，不以 `read_only: true` 声明代替检查。没有模型、规划、子 Agent、长期记忆、动态工具 resolver 或自定义结果策略。该限制不代表通用真实业务适配已经完成。

## API 顺序

前缀为 `/api/v1/agents/intent-synthetic/intent-sessions`：

1. 创建 session，再以 `expected_revision` 生成 `propose_task`，示例 task 为 `{"input":{"number":7}}`。
2. 核对提案并调用原 `confirm`；此时没有 Run、工具调用或工具审批。
3. 用户单独发起 `POST /{session_id}/proposals/{proposal_id}/execute`，正文与独立确认一样：`revision`、`request_digest`、`decision_digest` 和完整 `draft`。不接收幂等 key、审批、权限或 RunOptions。
4. 返回 202 与已接纳 Run。通过 `GET .../{proposal_id}/execution` 查询；尚未接纳时返回 `admitted:false`，不会自动创建任务。

执行只接受 `propose_task`。取消/替换提案不执行；解释和澄清不能确认。接纳后提案状态为 `executed`，不再允许用撤销选择接口暗示取消 Run。编辑需求、撤销选择或期限到达不会取消已接纳的 Run；原有 Run 控制与恢复仍走各自认证接口，本适配层不代办。

## 原子性、重放与故障

执行前刷新原凭据、身份与 ACL；经过准入 await 后再次刷新。SQL 准入事务内重查 session、owner、身份/凭据指纹、Agent 版本、会话/epoch、需求 revision、确切参数摘要、确认状态与期限；Run 参数从同一事务的实际持久行重读，预处理回调修改的行副本不能替代 SQL 证据。Run、提案消费和一对一映射共同提交，失败共同回滚。

准入、重放与查询均校验持久化 `input` 和实际执行使用的完整 `task_arguments`。预期工具参数由内置冻结 fixture 的任务声明和已确认输入重新派生；只保留原始输入、篡改派生参数也会被拒绝。准入拒绝共同回滚 Run、映射及提案消费，查询发现不一致返回 503。

幂等 key 由宿主根据 session/proposal 派生。已有 Run 的快速重放和 SQLite 准入竞态路径都重查绑定；不会收编同 key 的无映射请求。响应丢失时先查询，也可在原期限内重试同一 execute；只返回原 Run。提案过期后不再接受 execute，但 session 仍有效时可以查询已接纳结果。

检查点失败或 HTTP 等待方在 SQL 接纳后取消，可能保留一个 `initialization_incomplete` Run；查询会如实返回，重试不新建 Run、不自动恢复或批准工具。独立确认和执行记录都不能替代 RunEngine 的 Schema、scope、预算、当前工具授权或写审批。

底层 `AdmissionFence` 是可信宿主的 Python 回调：异步刷新必须返回 None，提交/重放回调只允许在原 SQL 事务中处理元数据，不能 await、commit、rollback、另开连接或产生外部副作用。它不进入 HTTP JSON。外部身份系统、注册表与 SQLite 之间不提供分布式原子撤权保证。

执行元数据单独使用 `agent_intent_execution_meta` schema 1 和 `agent_intent_executions`，不改变原提案 schema。绑定与提案记录一样保留，不自动清除。身份变化、session 过期或版本切换可能阻止旧提案查询；已有 Run 仍受标准运行器的身份与来源规则约束。

## 本地演示与验证

```powershell
python -m examples.intent_synthetic_execution
python -m pytest tests/test_intent_synthetic_execution.py tests/test_intent_execution_boundaries.py
```

演示应返回 `doubled:14`、一个 Run、一个逻辑工具操作、零模型调用、零工具审批和零长期记忆。测试使用合成身份、临时 SQLite、ASGITransport 与受控故障；双连接竞态测试只验证 SQL 原子性，不代表多实例部署验收。

本轮不增加 UI 执行按钮。真实浏览器/bfcache 仍受先前原点拒绝阻塞，没有重试或绕过；真实供应商、真实写工具、生产身份、云端与多实例验证均不在本阶段。
