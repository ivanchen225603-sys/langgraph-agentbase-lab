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

上述命令使用非 editable 安装，构建依赖按 pyproject.toml 在隔离环境安装。运行锁固定直接及传递包版本，未固定下载文件 hash；跨平台矩阵尚需 CI 实跑。配置模板是 `.env.example`，复制为 `.env` 可选，默认值已经足够。开发锁、测试、内部文档及验证脚本保留在本地开发目录，不纳入最小源码发布清单。

## 通用增强示例

```powershell
.\.venv\Scripts\python.exe examples\agentbase_enhancements.py
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
.\.venv\Scripts\python.exe examples\agentbase_enhancements.py
.\.venv\Scripts\python.exe -m scripts.demo
```

公开源码不包含开发者机器的运行证据、数据库、构建产物、测试套件或内部评估文档。运行时依赖检查可执行 `python -m pip check`；完整开发回归需要单独取得开发测试材料。

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

本清单提供可运行示例与依赖检查。完整开发测试套件及内部验证脚本不在公开文件清单中。源码发布许可证尚未指定，本次不附加许可证文件。
