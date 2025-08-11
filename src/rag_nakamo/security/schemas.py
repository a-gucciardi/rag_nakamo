from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, List

# pydantic schemas for security guard to normalize

HarmLabel = Literal["unharmful", "harmful"]
RefusalLabel = Literal["refusal", "compliance"]

class UserRequest(BaseModel):
    user_id: int
    user_prompt: str
    base_llm_response: str = ""

class ClassificationResult(BaseModel):
    """ Result of safety classification by the guard model with _call_classifier."""
    prompt_harm_label: HarmLabel
    response_refusal_label: RefusalLabel
    response_harm_label: HarmLabel

class GuardDecision(BaseModel):
    """ Decision made by the guard based on classification result with _decide.""" 
    status: Literal["allow","block","sanitize"]
    reason: str
    safe_message: Optional[str] = None
    classification: ClassificationResult

class GuardedResponse(BaseModel):
    decision: GuardDecision
    final_answer: str
    original_answer: str
    context_used: Optional[List[Dict[str, Any]]] = None