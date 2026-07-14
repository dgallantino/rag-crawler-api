"""Agent-facing retrieval endpoint: POST /v1/agent/search.

Called directly by LLM tool-calls. Tenant identity, top_k, rerank, and filters
are resolved/defaulted server-side. The agent can only supply query and an
optional collection slug.

"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.stubs import raise_not_implemented
from app.database import get_db
from app.schemas.query import AgentRetrievalResult, AgentSearchRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/search",
    response_model=AgentRetrievalResult,
    summary="Agent-facing retrieval search",
    operation_id="queryAgent",
)
def agent_search(
    body: AgentSearchRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> AgentRetrievalResult:
    raise_not_implemented()
