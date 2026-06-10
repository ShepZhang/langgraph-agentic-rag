# Resume Bullets

1. 构建基于 LangGraph 的 Agentic RAG 文档问答系统，将传统 retrieve-generate RAG 改造为状态机式 Agent workflow，包含 query rewriting、hybrid retrieval、reranking、retrieval grading、conditional retry、answer generation、citation verification 和 fallback 等节点。

2. 设计 BM25 sparse retrieval + dense vector retrieval + RRF fusion 的可配置 hybrid retrieval pipeline，使系统能够同时利用语义召回和关键词、文件名、缩写、编号等精确匹配信号。

3. 设计 naive RAG 与 Agentic RAG 的统一评估基础设施，使用同一批文档和同一套结构化问题集，对 answer correctness、context relevance、citation accuracy、fallback accuracy、unsupported claim count、latency、retry count 和 token-cost 字段进行对比。

4. 新增 ablation-study 框架，用可复现 JSON artifacts 和 Markdown report 跟踪 query rewrite、retrieval grading、retry/fallback、reranking 和 citation verification 等模块贡献，并明确区分当前 full-agentic proxy run 与后续可独立开关的真实 ablation。

5. 规划面向简历项目的后续升级路线，包括 structured query transformation、structured retrieval grading、full claim-level citation verification、trace logging、FastAPI service layer、workspace isolation 和 interactive evaluation dashboard。
