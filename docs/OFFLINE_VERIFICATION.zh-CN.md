# 有界离线交付验证

本入口仅验证 `scripts/offline_inputs.json` 明列的技术输入。它不运行私有 `scripts.verify` 或生命周期系统，不把离线通过记为原完整验证通过。通用功能已冻结，不为验证引入业务规则。

## 前提与运行

使用 Python 3.11–3.13、Node.js 22+，预先安装 `requirements-dev.lock` 的依赖。入口只读取已安装版本并执行 `pip check`；缺失或不匹配时失败，不联网安装。新环境安装依赖属于先行准备，不属于离线运行。

从最终候选提交创建干净导出，例如 PowerShell：

```powershell
$candidateCommit = git rev-parse HEAD
git archive --format=zip --output=.runtime-offline-source.zip $candidateCommit
Expand-Archive -LiteralPath .runtime-offline-source.zip -DestinationPath .runtime-offline-source
Set-Location .runtime-offline-source
..\.venv\Scripts\python.exe -m scripts.verify_offline --source-commit $candidateCommit --evidence-dir evidence/offline --build-dir dist/offline
```

目标导出及输出目录必须新建。也可使用只含清单文件的干净 checkout；不要在包含私有开发文件的原工作目录运行。`--source-commit` 是导出者提供的候选标识，入口不会伪称自行验证了远端 Git：导出者须保存真实提交、导出文件及报告摘要的对应证据。manifest 的 `development_baseline` 记录开发起点，不冒充后续候选提交。

## 固定门禁

1. 验证路径规范、symlink/外部输入、清单外文件、开发锁和窄范围敏感模式；不读取 `.env`。
2. 检查明确输入的 lint/格式、核心与已列技术示例的类型、技术 pytest、JUnit 和 coverage。
3. 运行客户端/SSE/取消/意图/UI 测试、六个技术示例、合成 RAG 评测、恢复演示和有界本地 ASGI 性能检查。
4. 离线构建 wheel/sdist，核对运行模块、资源和模板；仅将刚构建的本地 wheel 以 `--no-index --no-deps` 安装到临时目标。包外验证发现/恢复链，以及三个模板各 10 个离线评测。

保留总体至少 80%、`raglab` 至少 80%、`agentbase` 非零的原门槛。JUnit 拒绝重复用例、失败和错误；仅保留原先精确限定的 Windows symlink 权限 skip，并单独报告，不能计为通过。任一子检查失败、缺少报告、输入变化或产物不完整均不能通过。不会通过删除测试、降低阈值或跳过失败修饰结果。

## 输入与边界

白名单包含已有核心、选定技术测试及其直接 fixture/脚本、技术 catalog、合成评测数据和开发锁。私有文档/生命周期、`order_quality` 和其他业务示例不加入。旧私有交付流水线的两项专属测试不属于此入口；本入口的路径、证据与平台例外由独立技术测试覆盖。

旧 schema2 迁移测试使用两份合成 SQL 文本和原 provenance，不附带运行数据库。两份 SQL 按原始字节保存，`.gitattributes` 禁止换行转换，避免改变已记录的长度和摘要；不重新生成或改写历史证据来使测试通过。进程崩溃 fixture 仅操作新建临时目录中的模拟效果。

`verification.json` 绑定输入摘要、检查日志、唯一用例、覆盖率和构建包。`offline_gates_passed` 与 `full_verify_passed:false` 分开；包外模板评测与 Node 检查不累加进 pytest 数量。日志是临时合成记录，仍不得把它们当公开业务数据。

这不是 OS 网络沙箱。运行使用 local/scripted provider、关闭 tracing、清除模型凭据并禁止 pip 下载；技术测试可能使用本机 loopback TCP 验证 SSE，不进行真实浏览器操作。真实浏览器/bfcache 仍因原点拒绝而未验收，不换工具或云 browser 绕过。在线依赖审计、跨平台 CI、Docker、生产身份、供应商质量及多实例仍须各自证据，不由本入口推定通过。
