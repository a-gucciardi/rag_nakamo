from typing import List, Dict, Any, Optional
from openai import OpenAI
from rag_nakamo.security.prompt import CLASSIFIER_PROMPT
from rag_nakamo.security.schemas import ClassificationResult, GuardDecision, GuardedResponse
from rag_nakamo.settings import get_settings
import json, logging

logger = logging.getLogger(__name__)

BLOCK_MESSAGE = "This request was blocked by safety policies."
SANITIZE_MESSAGE = "Content was adjusted due to safety policies."

class PromptGuard:
    def __init__(self, max_context_chars: int = 10000):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.guard_model
        self.max_context_chars = max_context_chars

    def _build_context_snippet(self, docs: List[Dict[str, Any]]):
        """
        Turn top retrieved docs into a bounded-size snippet.
        Each doc dict expected to have 'content' and maybe 'source'.
        """
        parts = []
        total = 0
        for d in docs:
            piece = f"[Source: {d.get('source','?')}] {d.get('content','')}"
            if total + len(piece) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 0:
                    parts.append(piece[:remaining])
                break
            parts.append(piece)
            total += len(piece)
        return "\n---\n".join(parts)

    def _call_classifier(
        self,
        user_prompt: str,
        model_response: str,
        context_docs: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        context_snippet = self._build_context_snippet(context_docs or [])
        prompt = CLASSIFIER_PROMPT.format(
            user_prompt=user_prompt,
            model_response=model_response,
            context_snippet=context_snippet
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a JSON-only safety classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )

        raw_text = completion.choices[0].message.content.strip()

        # default to harmful if parsing fail
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Classifier returned invalid JSON; falling back to harmful defaults.")
            parsed = {
                "prompt_harm_label": "harmful",
                "response_refusal_label": "compliance",
                "response_harm_label": "harmful"
            }
        return parsed

    def _decide(self, classification: ClassificationResult) -> GuardDecision:
        # Policy:
        # 1. Harmful user prompt -> block
        if classification.prompt_harm_label == "harmful":
            return GuardDecision(
                status="block",
                reason="Harmful user prompt",
                safe_message=BLOCK_MESSAGE,
                classification=classification
            )
        # 2. Harmful model response AND not a refusal -> sanitize
        if (classification.response_harm_label == "harmful"
            and classification.response_refusal_label == "compliance"):
            return GuardDecision(
                status="sanitize",
                reason="Model produced harmful content without refusal",
                safe_message=SANITIZE_MESSAGE,
                classification=classification
            )
        # 3. Refusal or unharmful -> allow
        return GuardDecision(
            status="allow",
            reason="Compliant or refusal is safe",
            classification=classification
        )

    def classify_and_decide(
        self,
        user_prompt: str,
        draft_answer: str,
        context_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> GuardedResponse:
        raw = self._call_classifier(user_prompt, draft_answer, context_docs)
        classification = ClassificationResult(
            prompt_harm_label=raw.get("prompt_harm_label","harmful"),
            response_refusal_label=raw.get("response_refusal_label","compliance"),
            response_harm_label=raw.get("response_harm_label","harmful"),
        )
        decision = self._decide(classification)

        final_answer = draft_answer # if allowed, return original answer, else:

        if decision.status == "block":
            final_answer = decision.safe_message
        elif decision.status == "sanitize" and self.settings.sanitize:
            # brute-force sanitize (replace full answer).
            final_answer = decision.safe_message

        return GuardedResponse(
            decision=decision,
            final_answer=final_answer,
            original_answer=draft_answer,
            context_used=context_docs if context_docs else None
        )