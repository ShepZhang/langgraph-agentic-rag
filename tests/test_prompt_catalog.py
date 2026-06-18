from __future__ import annotations

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_REVISION_PROMPT,
    CITATION_VERIFICATION_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    RETRY_QUERY_REWRITE_PROMPT,
)
from prompting import (
    get_active_prompt_manifest,
    get_prompt_definition,
    get_prompt_template,
    render_prompt,
)
from prompting.catalog import _PROJECT_PROMPT_DEFINITIONS


EXPECTED_PROMPTS = {
    "agent.answer_generation": (
        ("current_query", "documents", "question"),
        "sha256:ef456ee86f56b4b61d2908d077d2d89f7b24995a18bd5ceb0240d86b3f370922",
    ),
    "agent.answer_revision": (
        ("answer", "documents", "question", "unsupported_claims"),
        "sha256:253006a5843c9a514299aec08520f2ea4f2ec57735f161dd6485d7a2bd81329a",
    ),
    "agent.citation_verification": (
        ("answer", "claims", "documents", "question"),
        "sha256:8c833a798f37b875561e16279e655bf393764cdebd42b2198f7f369ba1044eac",
    ),
    "agent.claim_extraction": (
        ("answer", "documents", "question"),
        "sha256:d249a0daf94d425fa2476efc4f2adabd55a42402bcf83f3bf1e869cb8e167eb2",
    ),
    "agent.claim_verification": (
        ("answer", "documents", "question"),
        "sha256:cc44c443a01e6f672dcacb2e9003f22a809742b54ee5b762d222ee17f4f00c17",
    ),
    "agent.query_rewrite": (
        ("chat_history", "question"),
        "sha256:3ccba60fa7582f1b319e3b07422ad569d212af7f49a7eb2a1c98872c5f32f4c7",
    ),
    "agent.query_transform": (
        ("chat_history", "question"),
        "sha256:24a29bac995a196aa1315c7515832e2cf5c14d4f0d8eacb53f5ec584a057e849",
    ),
    "agent.retrieval_grading": (
        ("current_query", "documents", "question"),
        "sha256:78c000ae92d4d549b8ef63800bd473df34adcbca88d52347648c531df086e004",
    ),
    "agent.retry_query_rewrite": (
        (
            "current_query",
            "documents",
            "grading_reason",
            "partial_relevance_context",
            "previous_queries",
            "question",
        ),
        "sha256:d80156327d23940a7e97c64409a08a1cce5b5e6ceb165dcc445ae42b0041363b",
    ),
    "tool.document_summary": (
        ("content", "max_points", "title"),
        "sha256:2c37f768a5fa0d1f01b5d7445c2fd04fd756f4f1c87549706b629216866ecde9",
    ),
}

ACTIVE_PROMPT_IDS = {
    "agent.answer_generation",
    "agent.answer_revision",
    "agent.citation_verification",
    "agent.claim_extraction",
    "agent.query_transform",
    "agent.retrieval_grading",
    "agent.retry_query_rewrite",
    "tool.document_summary",
}


def test_catalog_pins_prompt_ids_versions_variables_and_fingerprints():
    assert isinstance(_PROJECT_PROMPT_DEFINITIONS, tuple)
    assert {
        (definition.prompt_id, definition.version)
        for definition in _PROJECT_PROMPT_DEFINITIONS
    } == {(prompt_id, "v1") for prompt_id in EXPECTED_PROMPTS}

    for prompt_id, (variables, fingerprint) in EXPECTED_PROMPTS.items():
        definition = get_prompt_definition(prompt_id, version="v1")

        assert definition.version == "v1"
        assert definition.variables == variables
        assert definition.fingerprint == fingerprint

    assert get_active_prompt_manifest() == {
        prompt_id: {
            "version": "v1",
            "fingerprint": EXPECTED_PROMPTS[prompt_id][1],
        }
        for prompt_id in sorted(ACTIVE_PROMPT_IDS)
    }


def test_agent_prompt_constants_match_explicit_v1_catalog_templates():
    expected_constants = {
        "agent.query_rewrite": QUERY_REWRITE_PROMPT,
        "agent.retry_query_rewrite": RETRY_QUERY_REWRITE_PROMPT,
        "agent.retrieval_grading": RETRIEVAL_GRADING_PROMPT,
        "agent.answer_generation": ANSWER_GENERATION_PROMPT,
        "agent.claim_extraction": CLAIM_EXTRACTION_PROMPT,
        "agent.citation_verification": CITATION_VERIFICATION_PROMPT,
        "agent.answer_revision": ANSWER_REVISION_PROMPT,
        "agent.claim_verification": CLAIM_VERIFICATION_PROMPT,
    }

    for prompt_id, exported_constant in expected_constants.items():
        assert exported_constant == get_prompt_template(prompt_id, version="v1")


def test_query_transform_prompt_renders_current_structure():
    rendered = render_prompt(
        "agent.query_transform",
        chat_history="user: Discuss Agentic RAG.",
        question="How does it compare?",
    )

    assert rendered.startswith(
        "You transform user questions for private knowledge-base retrieval."
    )
    assert (
        '{"strategy": "rewrite", "rewritten_query": "standalone retrieval query", '
        '"expanded_queries": [], "sub_questions": [], "reason": "short reason"}'
        in rendered
    )
    assert "Chat history:\nuser: Discuss Agentic RAG." in rendered
    assert "Original question:\nHow does it compare?" in rendered


def test_document_summary_prompt_renders_exact_current_string():
    rendered = render_prompt(
        "tool.document_summary",
        max_points=2,
        title="Spec Notes",
        content="Grounded answers require retrieved evidence.",
    )

    assert rendered == (
        "Summarize the document using only the supplied text. "
        "Return at most 2 concise bullet points. "
        "Do not add unsupported facts.\n\n"
        "Title: Spec Notes\n\n"
        "Document:\n"
        "Grounded answers require retrieved evidence."
    )
