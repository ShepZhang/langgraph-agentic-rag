from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from evaluation.judge_parsing import ParsedJudgeResult, parse_semantic_judge_response


def _valid_payload() -> str:
    return json.dumps(
        {
            "semantic_correctness": {"score": 3, "reason": " accurate "},
            "groundedness": {
                "applicable": True,
                "score": 4,
                "reason": " grounded ",
            },
        }
    )


def test_parse_semantic_judge_response_valid_normal_case():
    result = parse_semantic_judge_response(_valid_payload(), fallback_triggered=False)

    assert result.raw_scores == {
        "semantic_correctness": 3,
        "groundedness": 4,
    }
    assert result.scores == {
        "semantic_correctness": 0.75,
        "groundedness": 1.0,
    }
    assert result.reasons == {
        "semantic_correctness": "accurate",
        "groundedness": "grounded",
    }


def test_parse_semantic_judge_response_valid_fallback_case():
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 4, "reason": " strong "},
            "groundedness": {
                "applicable": False,
                "score": None,
                "reason": " not applicable ",
            },
        }
    )

    result = parse_semantic_judge_response(payload, fallback_triggered=True)

    assert result.raw_scores == {
        "semantic_correctness": 4,
        "groundedness": None,
    }
    assert result.scores == {
        "semantic_correctness": 1.0,
        "groundedness": None,
    }
    assert result.reasons == {
        "semantic_correctness": "strong",
        "groundedness": "not applicable",
    }


@pytest.mark.parametrize("score", [True, 2.5, "3", -1, 5])
def test_parse_semantic_judge_response_rejects_invalid_semantic_scores(score):
    payload = json.dumps(
        {
            "semantic_correctness": {"score": score, "reason": "ok"},
            "groundedness": {"applicable": True, "score": 2, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"semantic_correctness\.score"):
        parse_semantic_judge_response(payload, fallback_triggered=False)


@pytest.mark.parametrize("score", [False, 1.5, "4", -1, 5])
def test_parse_semantic_judge_response_rejects_invalid_grounded_scores(score):
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 3, "reason": "ok"},
            "groundedness": {"applicable": True, "score": score, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"groundedness\.score"):
        parse_semantic_judge_response(payload, fallback_triggered=False)


@pytest.mark.parametrize("raw_text", ["", "   ", None])
def test_parse_semantic_judge_response_rejects_blank_or_non_string_input(raw_text):
    with pytest.raises(
        ValueError,
        match=r"raw_text must be a non-blank string; blank values are not allowed",
    ):
        parse_semantic_judge_response(raw_text, fallback_triggered=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw_text",
    [
        "```json\n{}\n```",
        'Here is the answer: {"semantic_correctness": {"score": 1, "reason": "ok"}, "groundedness": {"applicable": true, "score": 1, "reason": "ok"}}',
        "[]",
        '""',
        "1",
    ],
)
def test_parse_semantic_judge_response_rejects_fenced_prose_or_non_object_json(raw_text):
    with pytest.raises(ValueError):
        parse_semantic_judge_response(raw_text, fallback_triggered=False)


def test_parse_semantic_judge_response_rejects_extra_or_missing_top_level_keys():
    invalid_payloads = [
        {
            "semantic_correctness": {"score": 1, "reason": "ok"},
            "groundedness": {"applicable": True, "score": 1, "reason": "ok"},
            "extra": {},
        },
        {
            "semantic_correctness": {"score": 1, "reason": "ok"},
        },
        {
            "groundedness": {"applicable": True, "score": 1, "reason": "ok"},
        },
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError, match=r"top-level"):
            parse_semantic_judge_response(json.dumps(payload), fallback_triggered=False)


def test_parse_semantic_judge_response_rejects_extra_or_missing_nested_keys():
    invalid_payloads = [
        {
            "semantic_correctness": {
                "score": 1,
                "reason": "ok",
                "extra": 1,
            },
            "groundedness": {"applicable": True, "score": 1, "reason": "ok"},
        },
        {
            "semantic_correctness": {"reason": "ok"},
            "groundedness": {"applicable": True, "score": 1, "reason": "ok"},
        },
        {
            "semantic_correctness": {"score": 1, "reason": "ok"},
            "groundedness": {
                "applicable": True,
                "score": 1,
            },
        },
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValueError, match=r"semantic_correctness|groundedness"):
            parse_semantic_judge_response(json.dumps(payload), fallback_triggered=False)


@pytest.mark.parametrize(
    "semantic_value, grounded_value, field",
    [
        ("oops", {"applicable": True, "score": 1, "reason": "ok"}, "semantic_correctness"),
        (3, {"applicable": True, "score": 1, "reason": "ok"}, "semantic_correctness"),
        (
            [1, 2],
            {"applicable": True, "score": 1, "reason": "ok"},
            "semantic_correctness",
        ),
        (
            {"score": 1, "reason": "ok"},
            "oops",
            "groundedness",
        ),
        (
            {"score": 1, "reason": "ok"},
            3,
            "groundedness",
        ),
        (
            {"score": 1, "reason": "ok"},
            [1, 2],
            "groundedness",
        ),
    ],
)
def test_parse_semantic_judge_response_rejects_non_dict_nested_json_values(
    semantic_value,
    grounded_value,
    field,
):
    payload = json.dumps(
        {
            "semantic_correctness": semantic_value,
            "groundedness": grounded_value,
        }
    )

    with pytest.raises(ValueError, match=rf"{field} must be an object"):
        parse_semantic_judge_response(payload, fallback_triggered=False)


@pytest.mark.parametrize("applicable", [1, 0, "true", None])
def test_parse_semantic_judge_response_requires_grounded_applicable_to_be_bool(applicable):
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 2, "reason": "ok"},
            "groundedness": {"applicable": applicable, "score": 2, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"groundedness\.applicable"):
        parse_semantic_judge_response(payload, fallback_triggered=False)


def test_parse_semantic_judge_response_rejects_applicability_mismatch_for_nonfallback_false():
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 2, "reason": "ok"},
            "groundedness": {"applicable": False, "score": None, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"groundedness\.applicable"):
        parse_semantic_judge_response(payload, fallback_triggered=False)


def test_parse_semantic_judge_response_rejects_applicability_mismatch_for_fallback_true():
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 2, "reason": "ok"},
            "groundedness": {"applicable": True, "score": 2, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"groundedness\.applicable"):
        parse_semantic_judge_response(payload, fallback_triggered=True)


def test_parse_semantic_judge_response_rejects_non_null_grounded_score_when_unavailable():
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 2, "reason": "ok"},
            "groundedness": {"applicable": False, "score": 1, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"groundedness\.score"):
        parse_semantic_judge_response(payload, fallback_triggered=True)


@pytest.mark.parametrize("reason", ["", "   ", 1, None, True])
def test_parse_semantic_judge_response_rejects_blank_or_non_string_reasons(reason):
    payload = json.dumps(
        {
            "semantic_correctness": {"score": 2, "reason": reason},
            "groundedness": {"applicable": False, "score": None, "reason": "ok"},
        }
    )

    with pytest.raises(ValueError, match=r"semantic_correctness\.reason"):
        parse_semantic_judge_response(payload, fallback_triggered=True)

    payload = json.dumps(
        {
            "semantic_correctness": {"score": 2, "reason": "ok"},
            "groundedness": {"applicable": True, "score": 2, "reason": reason},
        }
    )

    with pytest.raises(ValueError, match=r"groundedness\.reason"):
        parse_semantic_judge_response(payload, fallback_triggered=False)


def test_parse_semantic_judge_response_deep_copies_input_maps_and_is_frozen():
    raw_scores = {"semantic_correctness": 3, "groundedness": 4}
    scores = {"semantic_correctness": 0.75, "groundedness": 1.0}
    reasons = {"semantic_correctness": "ok", "groundedness": "ok"}

    result = ParsedJudgeResult(raw_scores=raw_scores, scores=scores, reasons=reasons)

    raw_scores["semantic_correctness"] = 0
    scores["groundedness"] = 0.0
    reasons["semantic_correctness"] = "changed"

    assert result.raw_scores == {"semantic_correctness": 3, "groundedness": 4}
    assert result.scores == {"semantic_correctness": 0.75, "groundedness": 1.0}
    assert result.reasons == {"semantic_correctness": "ok", "groundedness": "ok"}

    with pytest.raises(FrozenInstanceError):
        result.raw_scores = {}


def test_parse_semantic_judge_response_accepts_whitespace_and_strips_reasons():
    payload = "\n  " + json.dumps(
        {
            "semantic_correctness": {"score": 3, "reason": "  neat  "},
            "groundedness": {
                "applicable": True,
                "score": 4,
                "reason": "\t grounded \n",
            },
        }
    ) + "  \n"

    result = parse_semantic_judge_response(payload, fallback_triggered=False)

    assert result.reasons == {
        "semantic_correctness": "neat",
        "groundedness": "grounded",
    }
