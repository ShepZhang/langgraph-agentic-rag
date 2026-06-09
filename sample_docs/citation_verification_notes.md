# Citation Verification Notes

Citation-aware generation means the answer must be grounded in the retrieved
chunks and cite the files that support each claim. A citation is useful only
when the cited source actually contains evidence for the sentence it supports.

Claim-level verification breaks an answer into individual claims and checks
each claim against the cited context. A supported claim is directly entailed by
the retrieved evidence. An unsupported claim is missing from the evidence,
contradicted by the evidence, or based on outside knowledge that was not in the
indexed documents.

When verification finds unsupported claims, the system should revise the answer
to remove or narrow those claims. If the remaining evidence is still too weak,
the system should fall back instead of producing an unsupported answer. This is
especially important for questions about credentials, payroll details, revenue,
executive biographies, or other facts not present in the sample corpus.

Citation verification improves reliability but adds latency and token cost. It
requires additional prompts for claim extraction and evidence checking, and it
may require a revision pass. Evaluation should track verification rate,
unsupported claim count, supported claim ratio, and whether fallback decisions
are correct.

Citation-sensitive questions should penalize answers that give the right topic
without a supporting source. For example, a correct answer about RRF should cite
retrieval pipeline notes, while a correct answer about claim verification should
cite citation verification notes.

