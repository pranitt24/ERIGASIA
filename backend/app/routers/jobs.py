from fastapi import APIRouter, Depends, HTTPException

from ..auth import current_user
from ..services import generation

router = APIRouter(dependencies=[Depends(current_user)])


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = generation.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found (the server may have restarted). Start generation again.")
    return {"status": j["status"], "step": j["step"], "steps": j["steps"], "error": j["error"]}
