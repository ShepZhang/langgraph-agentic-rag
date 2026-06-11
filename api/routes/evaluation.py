"""Evaluation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_evaluation_service
from api.schemas import EvaluationRunRequest, EvaluationRunResponse
from api.services.evaluation import EvaluationService


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationRunResponse)
def run_evaluation(
    request: EvaluationRunRequest,
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationRunResponse:
    """Run the lightweight evaluation pipeline."""

    return EvaluationRunResponse(
        **evaluation_service.run_evaluation(
            workspace_id=request.workspace_id,
            question_ids=request.question_ids,
            include_baseline=request.include_baseline,
        )
    )


@router.get("/{run_id}", response_model=EvaluationRunResponse)
def get_evaluation_run(
    run_id: str,
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationRunResponse:
    """Return a persisted evaluation run."""

    result = evaluation_service.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return EvaluationRunResponse(**result)
