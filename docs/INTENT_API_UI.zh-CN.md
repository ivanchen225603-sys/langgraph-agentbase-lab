# 无业务副作用的意图提案与独立确认

本文描述默认 API/UI。独立宿主显式启用的[合成本地只读执行闭环](INTENT_SYNTHETIC_EXECUTION.zh-CN.md)是另一个默认关闭的适配层，不改变本页确认本身的无执行行为。

本阶段接入 IntentDecision 的提案、查看、独立确认、撤销及需求失效流程。确认仅持久化用户选择，不创建 Run、不取消或恢复任务、不调用工具、不写记忆，也不写入工具审批表。所有响应均带 `execution_connected: false`；确认记录不是执行权限凭证。

## 启用方式与演示边界

RAG 教学宿主默认使用 `OfflineStructuredIntent`：按用户明确选择的类型和 JSON 输入生成固定结构化输出，不提供自然语言意图理解或真实模型质量保证。即使 RAG 模式另行选择了 OpenAI，这个提案演示也不会自动切换到真实模型。传入 `create_app(intent_interpreter=None)` 可关闭提案接口及其表安装。

独立宿主默认关闭，需要可信宿主显式启用：

```python
from agentbase.application import create_agent_app
from agentbase.intent_proposals import OfflineStructuredIntent, IntentProposalLimits

app = create_agent_app(
    runtime_factory,
    authenticate=authenticate,
    refresh_identity=refresh_identity,
    authorize_agent=authorize_agent,
    intent_interpreter=OfflineStructuredIntent(),
    intent_limits=IntentProposalLimits(proposal_seconds=120.0),
)
```

沿用现有认证、Agent ACL、主体解析和请求大小限制；没有新建账号、密钥、scope 或持续授权。页面入口为“意图提案”，与原直接创建/取消 Run 的页面分开。

## API 流程

以下路径前缀为 `/api/v1/agents/{agent_id}/intent-sessions`，`agent_id` 必须由宿主注册且对当前身份可见。示例任务 JSON 仍须满足该 Agent 的 `input_schema`。

1. `POST /`，正文 `{"conversation_id":"intent-demo"}`，由服务端签发 `session_id`、epoch 和 revision。
2. `GET /{session_id}` 获取最新会话版本。
3. `POST /{session_id}/proposals`，正文例如：

```json
{
  "expected_revision": 0,
  "draft": {
    "text": "请为此输入生成提案",
    "action": "propose_task",
    "task": {"input": {"text": "synthetic"}},
    "target_run_id": null
  }
}
```

`expected_revision` 是并发前置条件，不是权限来源。一次接受的需求推进 revision，并使旧提案失效；旧请求即使在网络中晚到也不能覆盖当前需求。替换和取消提案必须显式指定同主体、同 Agent、同会话、当前 epoch 的目标 Run。

4. `GET /{session_id}/proposals/{proposal_id}` 查看原文、提案、模式、版本、摘要、期限及状态。
5. `POST .../{proposal_id}/confirm` 记录一次独立确认；正文必须包含提案响应的 `revision`、`request_digest`、`decision_digest`，以及仍与原记录一致的完整 `draft`。
6. `POST .../{proposal_id}/revoke` 撤销待确认或已确认的选择，使用同样的精确绑定正文；保留原确认及撤销时间。
7. `POST /{session_id}/invalidate` 用于编辑需求时使旧版本失效；不接收业务执行参数。

没有 execute 接口。确认不替代未来写工具的精确审批，不扩大 scope，不复用旧计划或旧审批。生成/确认正文中额外的身份、权限、批准字段会被拒绝；业务 JSON 中的同名键仅是数据，不能用于鉴权。invalidate 仅使当前 session 的需求失效，不从正文取得任何权限。

## 隔离、期限与存储

- 身份来自可信宿主，`owner` 是现有主体/租户隔离键。跨租户部署必须由宿主提供唯一、可信的租户与主体映射；当前 local/静态 bearer 演示不等于生产 SSO 或多租户目录。
- 服务端同时绑定 owner、权限版本和排序后的 scopes 指纹、已验证凭据的指纹、Agent 发布绑定、会话、epoch、请求 ID 和需求 revision。UI/模型不能自报这些权限值。凭据指纹仅用于收窄访问，不保存原 bearer/cookie，也不作为可重放凭据返回。
- 提案响应使用 `Cache-Control: no-store`，并按 Authorization/Cookie 区分，避免缓存复用不同认证上下文的记录。独立宿主也执行同源检查。
- Cookie 变化可能保守地使旧 session 失效；当前指纹适配针对已有 bearer/local 宿主，不声称解决所有认证平台的会话映射。
- 同一主体/Agent/会话重新签发 session 会使原 session 及其待确认记录失效。不同 session、会话、主体、凭据、目标和需求版本的确认不能互换。
- 解释器 await 前后复检身份及 ACL；落库时在同步 SQLite 事务中重查 revision、目标和期限。确认/撤销没有跨 await 的事务，单次消费由写事务串行化。运行器关闭后拒绝晚到写入。
- 默认提案期限 120 秒、session 期限 1800 秒；期限只由宿主设置，服务端时钟是最终依据。浏览器计时器只辅助显示。过期状态按读取时刻计算，历史确认时间仍保留。
- 使用独立 `agent_intent_*` 表和版本 1 的启动迁移。记录包含需求原文、业务 JSON 及选择；过期不自动删除。默认每主体最多 16 个 session、128 个提案，全局分别 256/2048，包含过期记录；另有限制解释器耗时和并发。达到保留额度返回 429，保留/归档流程尚未实现。
- 这些检查不提供外部身份系统与 SQLite 的跨系统原子撤权保证；未来执行端必须再次按现有 RunEngine 和工具审批边界检查，不能直接消费此确认作为执行授权。

## 页面行为

生成提案后需重新勾选“已核对”才能点击独立确认。重复点击被页面抑制，服务端重复确认仍返回 409。需求、会话、Agent、身份变化会禁用旧确认，并使原需求失效；晚到响应不能替换新状态。确认响应丢失时显示“结果尚未核实”，可查询状态，不自动重试确认。页面只用文本方式展示提案。

查询暂时失败时保留原记录并显示“尚未核实”，允许再次查询，不发送需求失效请求。页面离开时释放本地请求占用、清除计时器和勾选状态；恢复时查询已有记录的服务端状态及期限，或提示重新生成。离开前的晚到响应不能更新恢复后的页面，也不会自动重发确认。

## 验证与当前剩余项

```powershell
python -m pytest tests/test_intent_proposals_api.py
python -m scripts.check_client
python -m mypy agentbase raglab
```

本阶段使用合成身份、离线解释器、临时 SQLite 和无外部副作用工具。测试涵盖版本/参数/凭据/会话/目标隔离、撤销、过期、跨连接消费竞争、权限撤销、关闭后响应、记录损坏和 UI 乱序。

真实浏览器验收尚未完成：浏览器安全策略明确报告，用户拒绝了 `http://127.0.0.1:18085` 的原点访问授权。本轮停止浏览器操作，没有换端口、换浏览器或使用其他方式绕过。Node DOM 测试和 HTTP 测试不能替代真实浏览器实证；需要用户允许该本机原点后继续。

执行端、真实模型、供应商及真实写工具均未接入。

初次实现（`50943de`）的自动化结果：349 个唯一 pytest 用例通过，其中新增 API/安全用例 45 个、既有相关回归 304 个；仅 Git 跟踪文件的独立导出通过 259 项，是上述用例的子集，不累加。前端共 18 个 Node 测试单元通过，其中新增 6 个模拟 DOM 用例。意图层三个模块行覆盖率 96.19%，不是全库覆盖率；mypy 58 个源文件、9 个修改相关 Python 文件的 Ruff/格式检查通过。wheel 与源码包构建及内容检查通过。未把此前 964、214、12 等基线数字相加。

后续 UI 恢复修复新增确定性测试：确认响应丢失后查询失败再成功、生成/确认/查询请求跨 `pagehide`/`pageshow` 的恢复，以及凭据切换。最终测试在原版 `50943de` 上出现 4 项预期失败，修复后通过。事件和 DOM 均为 Node 模拟，不代表已在真实浏览器验证往返缓存。该轮最终提交对应的回归、类型检查和构建记录位于本地 `evidence/intent-ui-recovery/verification.json`，与初次实现结果分别记录。

针对性安全复核为本轮自查及对抗性离线测试，并非独立人员/代理审查。已覆盖凭据/主体、session、会话、版本与目标边界，确认消费竞争、权限撤销、关闭后的晚到响应、过期写入边界、跨站请求、缓存隔离及未知确认结果；仍保留上文说明的生产身份、保留策略与执行接入限制。详细结果位于本地 `evidence/intent-api-v1/`，证据与运行数据库不纳入 Git。
