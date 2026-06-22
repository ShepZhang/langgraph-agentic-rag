from __future__ import annotations

import json

from evaluation.judge_evidence import (
    MAX_JUDGE_EVIDENCE_CHARS,
    MAX_JUDGE_EVIDENCE_CHUNKS,
    format_judge_citations,
    format_judge_evidence,
    select_judge_evidence,
)
from evaluation.schemas import EvaluationResult


def _empty_result() -> EvaluationResult:
    return EvaluationResult.empty(
        question_id="q001",
        question_type="single_doc",
        question="What is RAG?",
    )


def test_relevant_priority_deep_copy_and_exact_normalized_evidence_record():
    result = _empty_result()
    result.retrieved_documents = [
        {
            "source": "/workspace/retrieved.md",
            "page": 1,
            "chunk_id": "retrieved",
            "content": "retrieved evidence",
            "matched_queries": ["ignored"],
        }
    ]
    result.relevant_documents = [
        {
            "source": "/workspace/private/selected.md",
            "page": 7,
            "chunk_id": " c-1 ",
            "content": "  Line 1\nLine 2\t",
            "file_hash": "secret",
            "matched_queries": ["rag"],
        }
    ]

    selected = select_judge_evidence(result)
    selected[0]["matched_queries"].append("mutated")

    assert result.relevant_documents[0]["matched_queries"] == ["rag"]
    assert selected[0] is not result.relevant_documents[0]
    assert selected[0]["source"] == "/workspace/private/selected.md"
    assert format_judge_evidence(result) == (
        '[{"source":"selected.md","page":7,"chunk_id":"c-1","content":"Line 1 Line 2"}]'
    )


def test_source_path_only_records_do_not_emit_source():
    for field_name in ("source_path", "file_path"):
        result = _empty_result()
        result.relevant_documents = [
            {
                field_name: "/workspace/private/hidden.md",
                "content": "hidden evidence",
            }
        ]

        assert json.loads(format_judge_evidence(result)) == [
            {"content": "hidden evidence"}
        ]


def test_retrieved_fallback_when_relevant_is_empty():
    result = _empty_result()
    result.retrieved_documents = [
        {
            "source": "retrieved.md",
            "content": "fallback content",
        }
    ]

    assert select_judge_evidence(result) == [
        {
            "source": "retrieved.md",
            "content": "fallback content",
        }
    ]


def test_select_and_format_are_bounded_and_preserve_order():
    result = _empty_result()
    result.relevant_documents = [
        {
            "source": f"/workspace/doc{i}.md",
            "page": i,
            "chunk_id": f"chunk-{i}",
            "content": f"{i}" * (MAX_JUDGE_EVIDENCE_CHARS + 25),
        }
        for i in range(MAX_JUDGE_EVIDENCE_CHUNKS + 2)
    ]

    selected = select_judge_evidence(result)
    formatted = json.loads(format_judge_evidence(result))

    assert len(selected) == MAX_JUDGE_EVIDENCE_CHUNKS
    assert len(formatted) == MAX_JUDGE_EVIDENCE_CHUNKS
    assert [record["chunk_id"] for record in selected] == [
        f"chunk-{i}" for i in range(MAX_JUDGE_EVIDENCE_CHUNKS)
    ]
    assert [record["source"] for record in formatted] == [
        f"doc{i}.md" for i in range(MAX_JUDGE_EVIDENCE_CHUNKS)
    ]
    assert all(len(record["content"]) == MAX_JUDGE_EVIDENCE_CHARS for record in formatted)


def test_format_judge_citations_is_metadata_only_bounded_and_path_safe():
    result = _empty_result()
    result.citations = [
        {
            "source": rf"C:\private\case\doc{i}.md",
            "page": i,
            "chunk_id": f" c{i} ",
            "snippet": "  Alpha\nBeta\t",
            "file_hash": "secret",
            "score": 0.1,
            "matched_queries": ["rag"],
        }
        for i in range(MAX_JUDGE_EVIDENCE_CHUNKS + 1)
    ]

    formatted = json.loads(format_judge_citations(result))

    assert len(formatted) == MAX_JUDGE_EVIDENCE_CHUNKS
    assert formatted[0] == {
        "source": "doc0.md",
        "page": 0,
        "chunk_id": "c0",
        "snippet": "Alpha Beta",
    }
    assert all(
        set(record) == {"source", "page", "chunk_id", "snippet"} for record in formatted
    )


def test_windows_and_posix_paths_are_reduced_to_basenames():
    result = _empty_result()
    result.relevant_documents = [
        {
            "source": r"C:\Users\alice\Desktop\windows\alpha.txt",
            "content": "windows",
        },
        {
            "source": "/home/alice/docs/linux/beta.txt",
            "content": "posix",
        },
    ]
    result.citations = [
        {
            "source": r"C:\Users\alice\Desktop\windows\gamma.txt",
            "snippet": "windows citation",
        },
        {
            "source": "/home/alice/docs/linux/delta.txt",
            "snippet": "posix citation",
        },
    ]

    evidence = json.loads(format_judge_evidence(result))
    citations = json.loads(format_judge_citations(result))

    assert [record["source"] for record in evidence] == ["alpha.txt", "beta.txt"]
    assert [record["source"] for record in citations] == ["gamma.txt", "delta.txt"]


def test_url_sources_strip_userinfo_query_and_fragment_credentials():
    result = _empty_result()
    result.relevant_documents = [
        {
            "source": (
                "https://user:password@storage.example/private/doc.pdf"
                "?X-Amz-Credential=SECRET&X-Amz-Signature=SIG#private"
            ),
            "content": "evidence",
        }
    ]
    result.citations = [
        {
            "source": (
                "//token:secret@storage.example/private/citation.pdf"
                "?access_token=SECRET#fragment"
            ),
            "snippet": "citation",
        }
    ]

    evidence = json.loads(format_judge_evidence(result))
    citations = json.loads(format_judge_citations(result))

    assert evidence == [{"source": "doc.pdf", "content": "evidence"}]
    assert citations == [{"source": "citation.pdf", "snippet": "citation"}]
    serialized = json.dumps([evidence, citations])
    assert "SECRET" not in serialized
    assert "password" not in serialized
    assert "token:secret" not in serialized


def test_invalid_page_and_blank_metadata_are_omitted_and_non_string_content_becomes_empty():
    result = _empty_result()
    result.relevant_documents = [
        {
            "source": "   ",
            "page": True,
            "chunk_id": "  ",
            "content": 123,
        }
    ]
    result.citations = [
        {
            "source": "   ",
            "page": False,
            "chunk_id": "\n",
            "snippet": None,
        }
    ]

    evidence = json.loads(format_judge_evidence(result))
    citations = json.loads(format_judge_citations(result))

    assert evidence == [{"content": ""}]
    assert citations == [{"snippet": ""}]


def test_select_output_mutation_cannot_affect_result():
    result = _empty_result()
    result.retrieved_documents = [
        {
            "source": "notes.md",
            "content": "original",
            "matched_queries": ["rag"],
        }
    ]

    selected = select_judge_evidence(result)
    selected[0]["matched_queries"].append("mutated")
    selected.append({"content": "extra"})

    assert result.retrieved_documents == [
        {
            "source": "notes.md",
            "content": "original",
            "matched_queries": ["rag"],
        }
    ]


def test_formatting_is_compact_deterministic_and_preserves_unicode():
    result = _empty_result()
    result.relevant_documents = [
        {
            "source": "/workspace/资料/文件.md",
            "content": "  你好\n世界  ",
        }
    ]

    first = format_judge_evidence(result)
    second = format_judge_evidence(result)

    assert first == second == '[{"source":"文件.md","content":"你好 世界"}]'
