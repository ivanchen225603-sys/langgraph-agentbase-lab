# 通用记忆变更审计

`SqliteContextStore` 在现有 SQLite 事务内追加记忆变更证据。范围仅为通用 `agent_memories`；RAG 的 `memories` 表仍使用原有事件审计，本轮没有新增 HTTP 审计接口或模型工具。

## 写入与兼容

```python
# actor 必须来自可信宿主当前认证，不能从模型或请求正文重建。
record = contexts.create_memory(
    actor.owner, "assistant", "conversation", "synthetic preference",
    actor=actor,
)
contexts.update_memory(
    actor.owner, "assistant", "conversation", record.id, record.version,
    enabled=False, actor=actor,
)
contexts.delete_memory(
    actor.owner, "assistant", "conversation", record.id, actor=actor,
)
```

记录包含操作类型、owner/Agent/会话/记忆 ID、主体标识、旧新版本、变更前后摘要、来源引用指纹及时间。禁用和重新启用分别记录 `disable`、`enable`；到期和外部来源撤权不是存储变更，不伪造对应事件。

旧调用不传 `actor` 时仍然可用，标记为 `legacy-owner`，仅记录可信调用方传入的 owner，不补造身份、权限版本或认证结论。传入 `Identity` 时标记 `host-identity`，保留权限版本和身份指纹，并要求 `actor.owner == owner`。`Identity` 是数据契约，本身不证明认证；可信宿主仍须验证当前凭据、权限和来源。审计不授予执行权限。

增改删与审计必须共同提交；审计失败会回滚实际变更，版本冲突、配额拒绝和错误命名空间不会留下成功事件。删除保留原有语义，不自动恢复记忆。

## 迁移

宿主在启动迁移事务中调用 `SqliteContextStore.install(db)`，例如先执行 `BEGIN IMMEDIATE`。上下文 schema 从 1 升为 2，新增 `agent_memory_audit` 和命名空间索引，旧记忆、历史和会话内容不变。重复初始化不重复事件；旧数据不会补造历史审计。失败迁移回滚后可重试，不支持的版本在 DDL 前拒绝。

兼容旧安装器：连接尚未开启事务时，`install` 会先执行 `BEGIN IMMEDIATE`，但不自行提交或回滚，仍由调用方的连接上下文管理；已有事务不会被替换。

## 受限读取

```python
events = contexts.memory_audit(
    actor, "assistant", "conversation", memory_id,
    guard=lambda db: host_audit_read_allowed_now(db, actor, memory_id),
    after=0, limit=100,
)
```

`guard` 是可信宿主提供的同步函数，绑定当前认证、命名空间及审计读取策略，读取前后均必须明确返回 `True`。示例中的 `host_audit_read_allowed_now` 由宿主实现；始终返回 True 的占位函数不能用于不可信请求。没有新增默认 scope 或权限。读取固定为当前主体、一个 Agent、一个会话、一个记忆 ID，单页最多 200 条，按 `sequence` 游标翻页；没有跨租户全量接口。删除后读取历史仍须同样的审计授权。

## 数据与保留边界

审计不保存正文、原始 sources、scopes 或凭据；来源引用只保留 SHA-256 指纹，旧新记录也只保留摘要。指纹仍可能通过猜测或关联暴露信息，主体与命名空间标识也可能敏感，须按受限审计数据保护，不能当作匿名数据。

记录保留在同一数据库，不随记忆删除或 TTL 自动清除；本轮不增加归档、清理或审计配额，容量随变更增长，保留政策由宿主管理。日志不是防篡改账本，也不保证 SQLite 空闲页、WAL 或既有备份的物理擦除。摘要不能恢复正文，审计记录不能作为记忆、来源授权或恢复令牌。

## 验证

```powershell
python -m pytest tests/test_memory_audit.py
```

该测试完整纳入 Git，独立使用临时 SQLite 和合成身份，不依赖 ignored 测试。涵盖增改删/禁用、迁移与重复初始化、故障回滚、并发冲突、主体隔离、来源撤销、无正文审计及读取鉴权。完整验证和最终提交绑定记录位于本地 `evidence/memory-audit-v1/`。

Intent 执行端仍未接入；真实浏览器及 bfcache 验收继续受先前访问拒绝阻塞，本轮不重试。
