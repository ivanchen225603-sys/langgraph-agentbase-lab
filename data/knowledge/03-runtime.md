# LangGraph 运行与恢复

LangGraph 检查点是任务执行状态的权威来源，业务 SQLite 保存幂等接纳、操作意图、结果收据和可靠审计。

每次运行使用独立图线程，principal 与 run_id 的映射由服务端管理，客户端不能通过猜测 thread_id 越权读取检查点。

总 deadline 包含排队、配额等待、调用与退避，进程重启后恢复仍然使用最初的期限。

LangGraph interrupt 用于等待人工输入，取消通过 asyncio 任务取消传播，二者不是同一种机制。

检查点不保证外部系统 exactly-once；外部写成功但收据丢失必须进入 needs_reconciliation，禁止自动重放。

SQLite 适合单机教学且一次只能有一个写者，当前应用只支持单进程，不承诺十万 QPS 或生产 SLA。
