# Resume Bullets

1. 构建基于 LangGraph 的 Agentic RAG 文档问答系统，将传统 retrieve-generate RAG 改造为状态机式 Agent workflow，包含 structured query transformation、multi-query retrieval、hybrid retrieval、reranking、retrieval grading、conditional retry、answer generation、citation verification 和 fallback 等节点。

2. 设计 BM25 sparse retrieval + dense vector retrieval + RRF fusion 的可配置 hybrid retrieval pipeline，并接入 cross-encoder reranker 的 candidate top-k / final top-n 配置，使系统能够同时利用语义召回、精确关键词匹配和候选重排序。

3. 设计 naive RAG 与 Agentic RAG 的统一评估基础设施，使用同一批文档和同一套结构化问题集，对 answer correctness、context relevance、citation accuracy、fallback accuracy、unsupported claim count、latency、retry count 和 token-cost 字段进行对比。

4. 新增 ablation-study 框架，用可复现 JSON artifacts 和 Markdown report 跟踪 query transformation、retrieval grading、retry/fallback、reranking 和 citation verification 等模块贡献；evaluation artifacts 记录脱敏 runtime config，便于复现实验配置和对比 latency/cost trade-off。

5. 规划面向简历项目的后续升级路线，包括 structured retrieval grading、decomposition sub-question retrieval、full claim-level citation verification、trace logging、FastAPI service layer、workspace isolation 和 interactive evaluation dashboard。
