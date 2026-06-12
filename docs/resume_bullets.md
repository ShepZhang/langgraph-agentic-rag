# Resume Bullets

1. 构建基于 LangGraph 的 Agentic RAG 文档问答系统，将传统 retrieve-generate RAG 改造为状态机式 Agent workflow，包含 structured query transformation、multi-query retrieval、hybrid retrieval、reranking、structured retrieval grading、partial-relevance recovery、answer generation、claim-level citation verification、answer revision 和 fallback 等节点。

2. 设计 BM25 sparse retrieval + dense vector retrieval + RRF fusion 的可配置 hybrid retrieval pipeline，并接入 cross-encoder reranker 的 candidate top-k / final top-n 配置，使系统能够同时利用语义召回、精确关键词匹配和候选重排序。

3. 设计 naive RAG 与 Agentic RAG 的统一评估基础设施，使用同一批文档和同一套结构化问题集，对 answer correctness、context relevance、citation accuracy、fallback accuracy、unsupported claim count、latency、retry count 和 token-cost 字段进行对比。

4. 实现 claim-level citation verification：对 draft answer 抽取原子 claims，逐条验证 claim 是否被 cited chunks 支持；对 unsupported 或 partially supported claims 触发一次 answer revision，修订后仍失败则 fallback，降低 unsupported answer 和 wrong citation 风险。

5. 设计 typed Tool Registry 与依赖注入边界，统一 retriever、claim citation verifier、document summary 和 safe calculator 的参数校验、执行结果、错误语义与 trace diagnostics；将检索和引用验证节点接入 Registry，同时保持非自主规划的可扩展 Agent 工具架构。

6. 新增可执行的 V0-V6 cumulative ablation-study 框架、本地 JSONL trace logging、FastAPI 服务层和 workspace-aware retrieval isolation，用可复现 JSON artifacts、Markdown report、节点级运行轨迹和 HTTP API 跟踪 query transformation、structured retrieval grading、conditional retry/fallback、hybrid retrieval、reranking 和 citation verification 的增量贡献；evaluation artifacts 记录脱敏 runtime config，便于复现实验配置和对比 latency/cost trade-off。
