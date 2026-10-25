# IntentDecision 意图提案层

`agentbase.intent` 是可选的宿主适配模块，用于在进入执行器之前检查结构化意图提案。它不创建 Run、不调用工具、不批准操作、不取消任务、不写记忆，也不改变现有 API、UI 或默认工作流。

## 五种输出

| action | 内容 | review 结果 |
|---|---|---|
| `respond` | 解释或回答 | `respond`，无提案 |
| `clarify` | 请求补充信息 | `clarify`，无提案 |
| `propose_task` | 新任务的 `task.input` | `proposal`，等待宿主处理 |
| `propose_replace` | 新任务输入、原 Run ID 和原输入摘要 | `proposal`，等待宿主处理 |
| `propose_cancel` | 原 Run ID 和原输入摘要 | `proposal`，等待宿主处理 |

`task` 包装对象的 `input` 可以是任意 JSON，包括 null；它仍须经过目标 Agent 的业务 Schema 校验。`owner`、`scopes`、`approval` 等控制字段不在提案 Schema 中，额外字段会被拒绝。业务输入中的同名键只是业务数据，不能用来构造身份或授权。

## 宿主接入顺序

```python
from uuid import uuid4

from agentbase.contracts import Message
from agentbase.intent import (
    IntentContext, IntentDecision, IntentSession, read_intent_target,
)

# identity、epoch、agent_id、conversation_id 均来自已认证宿主状态。
context = IntentContext(
    request_id=uuid4().hex,
    identity=identity,
    agent_id=agent_id,
    conversation_id=conversation_id,
    epoch=epoch,
)
# 只有当前用户明确指向某个旧任务时，宿主才读取并绑定 target。
target = read_intent_target(engine.store, identity, target_run_id) if target_run_id else None
session = IntentSession(
    context,
    Message(role="user", content=user_text),
    origin="authenticated-user",
    target=target,
)
model_input = session.model_input()  # 作为低信任用户数据，不拼接为系统指令。
output_schema = IntentDecision.model_json_schema()

# 使用宿主已有 provider 获得 JSON 对象；此模块不运行另一套模型循环。
# provider await 返回后，宿主重新认证并读取最新 request_id、epoch 和目标。
review = session.review(
    parsed_model_json,
    current=fresh_context,
    target=fresh_target,
)
```

`model_input()` 包含请求 ID、绑定摘要、用户原文，以及可选目标 ID/输入摘要，不包含宿主身份和权限。宿主应保存每次请求的 session，给新请求分配新 ID；不能根据模型返回内容重建 session。`current` 必须来自最新宿主状态，不能直接复用 await 前的快照冒充刷新。

## 校验与执行边界

- `interpretation` 为 `negated`、`ambiguous` 或 `conflicting` 时返回澄清。格式错误、非 JSON、过大、循环或过深数据也返回通用澄清，不回显原始错误内容。
- 对动作提案额外检查常见中英文否定表达，即使模型声称 `explicit` 也可否决。此规则保守且不完备，可能误拦“不要忘记”等表达，不能当作通用语义识别或真实模型准确率证明。
- 来源标签由宿主决定。文档、记忆、子 Agent 输出、非 user 角色或附带来源引用的消息不能作为直接用户控制请求。不得把这些内容重新标记为 `authenticated-user`。
- 每个 session 只审查一次，包括拒绝结果。请求 ID、身份权限快照、Agent、会话或 epoch 变化会拒绝旧响应。替换/取消还必须提供与最初绑定一致的最新目标快照。
- 目标快照的 `active` 仅检查取消标记、最终回执和当前 epoch；不代替运行器的期限、暂停审批时钟或检查点状态判断，避免在提案层另写一套执行政策。
- 单次消费是进程内约束，不提供跨进程持久化提案账本。重启后丢弃旧 session；如需持久恢复，宿主另行保存认证请求与消费状态，不能把返回 JSON 当作可重放权限凭证。
- `proposal` 始终是不可信提案。宿主应展示原文、目标和新需求，取得独立明确确认；确认必须绑定确切的新输入和目标，并在执行时重新校验身份、Agent ACL、业务 Schema 和最新状态。不能把模型的 `explicit` 当成用户批准。
- 确认后仍使用已有 `RunEngine.submit/cancel`。`RunOptions(switch_topic=True)` 按整个会话推进 epoch，会停止该会话的旧任务，不是只取消提案中的一个 Run；有多个旧任务时，宿主必须说明替换范围。不得继承旧计划、旧工具审批或根据未知写结果盲目重提。
- 提案检查与后续执行之间没有跨操作原子性保证；并发确认时需宿主在实际控制事务中重查绑定。此模块没有自动确认、执行或 HTTP 接入入口。

## 离线验证

```powershell
python -m pytest tests/test_agentbase_intent.py tests/test_agentbase_intent_runtime.py
python -m ruff check agentbase/intent.py tests/test_agentbase_intent.py tests/test_agentbase_intent_runtime.py
python -m mypy agentbase/intent.py
```

测试使用结构化合成输出、临时 SQLite、真实 RunEngine 和假写工具。覆盖无副作用澄清/提案、源文档与子 Agent 冒充控制、旧响应和目标失效、跨会话目标，以及执行中替换后必须重新审批；不调用真实模型或外部服务。
