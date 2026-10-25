# LangGraph Agent Base Lab

可运行的 Python 通用 Agent 基座与 RAG 教学项目：LangChain 组件组织文档、检索和工具，LangGraph 执行入库图、并行 RAG 子图、DAG Agent 与持久恢复。提供可配置 Agent、工具审批、HTTP/SSE、Web 页面和独立业务示例。默认使用词法哈希向量和证据原句抽取，**不是神经语义模型，也不是公司共享或生产系统**。仓库只包含源码、文档及合成教学数据。

## 安装与启动

在 PowerShell 中执行；本机实际验证 Python **3.11.9**。依赖安装需要网络，核心演示运行无需模型密钥。使用独立虚拟环境，不依赖全局包。

```powershell
# 先进入下载或克隆得到的项目根目录。
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps .
.\.venv\Scripts\python.exe -m raglab
```

完成安装后可直接运行最后一条命令。页面：<http://127.0.0.1:8001>，交互 API：<http://127.0.0.1:8001/docs>。首次点击“建立示例索引”，再提问“RAG 为什么要把文档分块？”。建立前 `/health/ready` 返回 503 是预期行为，`/health/live` 返回 200。按 Ctrl+C 停止。一个数据目录只允许一个服务进程，**不要添加多个 Uvicorn worker**。

上述命令使用非 editable 安装，构建依赖按 pyproject.toml 在隔离环境安装。运行锁固定直接及传递包版本，未固定下载文件 hash；跨平台矩阵尚需 CI 实跑。配置模板是 `.env.example`，复制为 `.env` 可选，默认值已经足够。Git 按明确清单跟踪开发锁、技术测试与验证入口；最小源码分发包仍只包含运行所需材料，私有文档及生命周期记录不随之发布。

## 最小 Agent 接入

可直接从安装包生成 Agent 起步项目，模板包含 Agent 配置、样例输入、10 个评测用例和独立运行入口：

```powershell
.\.venv\Scripts\python.exe -m agentbase.scaffold extraction ./my-extractor
.\.venv\Scripts\python.exe ./my-extractor/run.py --cases ./my-extractor/cases.jsonl
```

`extraction` 从合成技术记录中抽取显式标注的 `record_id`、`label`、`enabled`；缺失或无法识别的字段返回 `null`，不推断缺失值。另两类模板为 `research`（证据工具与引用输出）和 `confirmed-action`（精确确认后保存本地记录）。目标目录必须不存在，脚手架不会覆盖已有项目。默认离线 fixture 只验证工程流程；它不是专业语义质量证明。提取/研究模板显式 `--mode openai --model YOUR_MODEL` 后使用环境中的 `OPENAI_API_KEY`，本轮未执行付费模型。确认动作模板使用固定流程，不需要模型；评测器只检查待审批状态，绝不自动批准。

新宿主可以用 `build_registry` 注册工作流及可信工具/模型，再通过 `async with open_runtime(registry, data_dir=..., identity_resolver=...) as engine` 管理数据库迁移、检查点、独占锁、启动与关闭。独立 HTTP 服务使用 `create_agent_app(runtime_factory, authenticate=..., refresh_identity=...)`，无需 RAG Settings/索引；认证及刷新回调必须由宿主提供。启动/关闭异常不会授予匿名权限，同目录只运行一个实例。

`function_tool(async_function, name=..., scope=..., read_only=...)` 根据类型/Pydantic 模型生成 Schema；`context_parameter` 显式注入 RunContext，运行上下文不能出现在模型入参或返回类型中。`langchain_tool` 支持提供原生 coroutine 的 LangChain 工具，要求明确输出类型，拒绝同步线程回退、错误抑制和工具 callbacks。写入仍须声明幂等/收据契约，并经过现有审批和 ToolExecutor；自动 Schema 不代表自动授权。

通用评测入口为 `agentbase.evaluation.load_cases/evaluate_cases`，支持精确结果、包含/不包含、Schema、工具调用和已应用操作数量检查。成功样例必须有业务断言；中间状态、不可见结果、超时和检查异常不能误报通过。`diagnose_run` 默认仅返回事件类型、状态、操作阶段和预算元数据，不返回正文、提示词或工具参数；未知模型用量明确保留 unknown，不当成零费用。样例的运行元数据仍保存在本地目录，诊断中不含费用金额。

安装完成后，可先运行不需要模型密钥的最小示例：

```powershell
.\.venv\Scripts\python.exe -m examples.agentbase_minimal
```

示例在临时目录中注册两个工具、绑定声明式 fixed 配置并运行同一个 RunEngine，验证依赖结果传递和幂等重放，结束后清理数据。接入新功能时替换 handler 和 Schema/配置即可；需要模型时注册 provider 并选择 conversation，需要动态子 Agent 编排时显式选择 planning。无需新增执行器，也不要求每个 Agent 都使用模型。

宿主可通过 `SqliteContextStore(connection, memory_limits=MemoryLimits(...))` 启用记忆存储配额；省略参数保留原行为。默认配额为每个 `(owner, agent_id, conversation_id)` 1000 条/8 MiB、每个 owner 10000 条/64 MiB，统计完整 UTF-8 JSON payload，禁用和过期记录仍计费，删除释放容量。停用或缩小正文始终允许（缩小按 JSON 编码后的正文大小判断），版本与时间字段可能增加少量计费字节；这些额度不代表 SQLite 文件大小上限。启用配额要求数据库编码为 UTF-8。检索 `max_memories` 另行限制进入上下文的记忆数量。

`engine.submit(identity, agent_id, request_key, value)` 的第三参数是幂等请求 key，owner 来自 identity；轮询使用 `engine.view(identity.owner, run["run_id"])`。不同请求使用不同 key，重试同一请求使用原 key。历史 Run 按旧绑定执行：发布新 Agent 版本时保留旧绑定，重启时使用 registry.restore 装配所需历史版本。

客户端丢失 Run ID 时，可用 `GET /api/v1/agents/{agent_id}/runs?limit=20` 查询当前主体的运行索引，再以 `next_after` 分页。索引只给 ID，不执行或恢复任务；状态与结果仍走原详情接口。排序及鉴权边界见[通用运行索引](docs/RUN_DISCOVERY.zh-CN.md)。

通用 Agent 页面也可刷新/翻页发现运行，并明确查询所选记录；可恢复且无待审批的 Run 使用独立普通恢复按钮，点击后重新核对状态。选择记录不会执行，普通恢复不代替精确审批；页面恢复或身份变化后需重新查询。当前验证为离线 DOM/ASGI，真实浏览器验收仍未完成。

## 通用增强示例

```powershell
.\.venv\Scripts\python.exe -m examples.agentbase_enhancements
```

示例用确定性 provider 和临时 SQLite 数据运行两个中立 Agent：可选规划提交两波任务，单 worker 持久派发与等待；长上下文生成有来源的工件，子结果继续携带输入来源。输出应为 succeeded、2 个子任务、3 次规划请求及额外摘要请求。它验证工程流程与来源追溯，模型质量由接入者单独评测。

planning、delegation 和 context.management 均需显式配置；未配置时保留 conversation/fixed 的原行为。运行库 schema8、工件 schema2；文档/提案之外的真实文件与 shell 由宿主按需要提供工具适配。RunEngine 是唯一执行器，子任务沿用工具授权、审批、祖先预算和固定期限；工件与运行状态必须使用同一 SQLite 事务连接。宿主 Agent ACL 应在 start 前配置，外部异步 ACL 不提供数据库事务级原子撤权。

## 三个可复现演示

```powershell
.\.venv\Scripts\python.exe -m scripts.demo
```

该脚本使用全新临时目录，退出清理；结果在 `evidence/demos.json`。

1. 引用问答：分块问题返回 `answered`、真实 chunk/source/index_version 和完整原句；无资料问题拒答。
2. 多轮与记忆：先问 Orion 备份，再问“它的超时时间是多少？”，解析成 Orion 并回答 8 秒；显式记忆写入、纠正到 v2、召回。长期记忆通过独立工具/API 查询，不默认注入知识问答或自动永久记忆。
3. 多任务：显式 a→b 依赖把前驱的结构化答案与来源传给后继；执行前 `interrupt` 暂停，批准后恢复成功；另一运行取消后终态不能恢复。

UI 对应原 RAG 四页及通用 Agent 页：证据问答、长期记忆、任务与恢复、运行诊断。任务 JSON 编辑器允许明确 DAG；分号或换行只拆独立问题，服务不猜测业务写权限。接口连接断开不会取消已接纳的后台 Agent；使用 cancel 接口明确取消。

## 运行检查

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m examples.agentbase_enhancements
.\.venv\Scripts\python.exe -m scripts.demo
```

最小源码分发包不包含开发者机器的运行证据、数据库、构建产物、测试套件或内部评估文档。Git 仓库额外跟踪明确列出的通用技术测试、合成评测与开发锁；私有开发材料和业务示例仍保留本地。运行时依赖检查可执行 `python -m pip check`。

## 模式与边界

| 模式 | 实际组件 | 验证范围 | 外部依赖 |
|---|---|---|---|
| local 默认 | LC Embeddings、Document、RecursiveCharacterTextSplitter、BaseRetriever、RunnableLambda、StructuredTool；真实 StateGraph/Send/interrupt/AsyncSqliteSaver | 本机流程、API、故障、安全和 UI | 安装后无需模型服务；词法同义表达、复杂推理能力有限 |
| openai 显式选择 | OpenAIEmbeddings + ChatOpenAI + ChatPromptTemplate + 同一图 | 构造与替代 provider 的离线模拟；**没有真实模型端到端调用** | 网络、密钥、服务可用性、数据外发、按量费用；公司数据及付费调用需另行授权 |

向量存储是 SQLite 持久数据 + 当前版本的 Python 精确扫描，BM25 每次按授权快照构建；适合教学小数据。重排是词法规则，语义缓存关闭。OpenAI 模式仍有词法相关性门槛与保守完整原句验证，自由改写会回退。引用合法只保证这段文本来自授权证据，不能证明资料本身正确或跨句条件全部满足。PDF/OCR、SSO、向量数据库、GPU 本地模型、分布式队列未实现。

Dockerfile 和 Compose 配置可用于容器化运行。Compose 仅发布到宿主机 loopback，容器内部使用静态 bearer 身份演示。容器构建与运行仍需在具备 Docker 的环境中自行验证。

## 项目入口

运行步骤和关键边界均在本 README 中。交互 API、OpenAPI Schema 与接口文档随服务提供：`/docs` 和 `/openapi.json`。通用 Agent 的启动入口为 `examples/agentbase_server.py`，离线双 Agent 演示为 `examples/agentbase_enhancements.py`，RAG、记忆与任务演示为 `python -m scripts.demo`。

运行后的 API 提供 OpenAPI 文档。

## 通用 Agent 基座

新旧接口共用同一 RunEngine；配置支持模型工具循环、固定依赖流程、上下文/记忆、Prompt、Schema、工具与额度，公共 HTTP/SSE 和通用页面已接入。启动：`python -m examples.agentbase_server --data-dir .runtime-agentbase-demo --port 8001`。可从该入口、`examples/agentbase.json` 与 `examples/agentbase_enhancements.py` 开始接入新 Agent。

本清单提供可运行示例、依赖检查与有限集成验证。完整内部测试套件和验证脚本不在公开文件清单中。源码发布许可证尚未指定，本次不附加许可证文件。

## 有限集成验证

Git checkout 可在独立虚拟环境安装开发依赖并运行已跟踪的测试：

```powershell
python -m pip install -e '.[dev]'
python -m pytest $(git ls-files 'tests/*.py' | Where-Object { $_ -match '/test_' })
python -m scripts.check_client
```

前端检查需要 Node.js 22 或以上版本。新增取消观察测试与原 UI 测试使用独立 DOM/时钟环境，保留双方断言。离线票据适配器仅是显式合成示例，未接入生产默认路径。`IntentDecision` 默认 HTTP/UI 的确认仅记录选择，执行端不接入。独立宿主可显式启用[合成本地只读执行闭环](docs/INTENT_SYNTHETIC_EXECUTION.zh-CN.md)，仅运行内置算术 fixture；`python -m examples.intent_synthetic_execution` 在临时目录与进程内 ASGI 中演示，不调用真实模型、写工具或供应商。

接入与边界见 [取消与收尾](docs/CANCELLATION.md)、[意图提案](docs/INTENT_PROPOSALS.zh-CN.md)、[离线票据示例](docs/OFFLINE_TICKET_RECEIPTS.md)；本轮整合的实际验证范围见 [整合验收](docs/INTEGRATION_ACCEPTANCE.zh-CN.md)。

提案/确认接口、期限、隔离与当前验收状态见 [意图提案 API/UI](docs/INTENT_API_UI.zh-CN.md)。教学页面使用固定结构化演示，不声称真实自然语言意图理解能力。

Git checkout 的统一离线门禁见[有界离线交付验证](docs/OFFLINE_VERIFICATION.zh-CN.md)。预先安装开发锁后，从新建的干净候选导出运行 `python -m scripts.verify_offline --source-commit COMMIT --evidence-dir evidence/offline --build-dir dist/offline`；入口不下载依赖，保留 80% 覆盖率门槛，并核对包外运行。它不等于原私有完整流水线，也不代表真实浏览器、在线审计或部署已验收。
