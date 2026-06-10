# Resume Bullets

1. 构建基于 LangGraph 的 Agentic RAG 文档问答系统，将传统 retrieve-generate RAG 改造为状态机式 Agent workflow，包含 query transformation、retrieval、retrieval grading、conditional retry、answer generation、citation verification 和 fallback 等节点。

2. 设计 naive RAG 与 Agentic RAG 的统一评估基础设施，使用同一批文档和同一套结构化问题集，对 answer correctness、context relevance、citation accuracy、fallback accuracy、unsupported claim count、latency、retry count 和 token-cost 字段进行对比。

3. 新增 ablation-study 框架，用可复现 JSON artifacts 和 Markdown report 跟踪 query rewrite、retrieval grading、retry/fallback、reranking 和 citation verification 等模块贡献，并明确区分当前 full-agentic proxy run 与后续可独立开关的真实 ablation。

4. 扩充 reliability-oriented evaluation dataset 至 36 条结构化问题，覆盖 single-doc、多 chunk 综合、ambiguous、unanswerable、distractor、comparison、follow-up、citation-sensitive、cross-file 和 false-premise 场景。

5. 规划面向简历项目的后续升级路线，包括 hybrid retrieval、reranker、structured retrieval grading、claim-level citation verification、trace logging、FastAPI service layer、workspace isolation 和 interactive evaluation dashboard。
