import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI

from rag_nakamo.settings import get_settings
from rag_nakamo.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ResponseAgent(BaseAgent):
    """
    Final answer formatter and validator for regulatory content.

    Expects a message dict with:
      question: str
      answer | final_answer | output: str
      retrieved_docs | docs | context_docs: List[{content, source, ...}]

    Output:
      {
        decision: "approve" | "revise",
        formatted_answer: str,
        metrics: { coverage, total_claims, unsupported },
        unsupported_claims: [...],
        llm_assessment: {...}  # optional
      }
    """

    def __init__(
        self,
        name: str = "ResponseAgent",
        description: str = "Final response formatting and validation agent",
        min_coverage: float = 0.7,
        max_unsupported: int = 2,
        enable_llm_assessment: bool = True,
        enable_regulatory_formatting: bool = True,
    ):
        super().__init__(name, description)
        settings = get_settings()
        self.model = settings.response_model
        self.validation_model = getattr(settings, "validation_model", self.model)
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.min_coverage = min_coverage
        self.max_unsupported = max_unsupported
        self.enable_llm_assessment = enable_llm_assessment
        self.enable_regulatory_formatting = enable_regulatory_formatting

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format the final answer."""
        logger.info(f"Processing message with ResponseAgent: {self.name}")
        
        # Extract question
        question = message.get("question", "")
        
        # Normalize answer - handle both string and dict formats
        if isinstance(message, str):
            answer = message
        else:
            answer = (
                message.get("answer")
                or message.get("final_answer")
                or message.get("output")
                or message.get("content", "")
            ).strip()

        # Normalize docs - handle various doc formats from your pipeline
        docs = []
        for key in ["retrieved_docs", "docs", "context_docs", "security_docs"]:
            docs = message.get(key, [])
            if docs:
                break
        
        # Handle case where docs might be a string or single dict
        if isinstance(docs, str):
            docs = [{"content": docs, "source": "unknown"}]
        elif isinstance(docs, dict):
            docs = [docs]

        if not answer:
            return self._error_result("Missing answer content")

        # Format the answer using LLM if available and enabled
        if self.enable_regulatory_formatting and self.client:
            formatted_answer = self._format_answer_with_llm(answer, question, docs)
        else:
            formatted_answer = self._format_answer_basic(answer, question, docs)
        
        # Validate if docs are available
        if docs:
            claims = self._extract_claims(formatted_answer)
            supported, unsupported = self._check_support(claims, docs)
            coverage = (len(supported) / len(claims)) if claims else 1.0
            decision = self._make_decision(coverage, unsupported)
        else:
            claims = []
            supported = []
            unsupported = []
            coverage = 1.0  # No docs to validate against
            decision = "approve"

        # LLM assessment
        assessment = {}
        if self.enable_llm_assessment and self.client and docs:
            assessment = self._get_llm_assessment(question, formatted_answer, claims, supported, docs)

        result = {
            "decision": decision,
            "formatted_answer": formatted_answer,
            "metrics": {
                "coverage": coverage,
                "total_claims": len(claims),
                "unsupported": len(unsupported),
            },
            "unsupported_claims": unsupported,
            "llm_assessment": assessment,
        }
        
        logger.info(f"ResponseAgent decision: {decision} (coverage: {coverage:.2f})")
        return result

    def _format_answer_with_llm(self, answer: str, question: str = "", docs: List[Dict] = None) -> str:
        """Format the answer using LLM with regulatory expert prompt."""
        try:
            # Prepare document context for the LLM
            doc_context = ""
            if docs:
                doc_context = "\n\nREGULATORY DOCUMENTS:\n"
                for i, doc in enumerate(docs, 1):
                    source = doc.get('source', f'Document_{i}')
                    content = doc.get('content', '')
                    doc_context += f"\n[{source}]\n{content}\n"
            
            regulatory_prompt = """You are a regulatory expert assistant. 
Your task is to provide comprehensive, accurate answers to regulatory questions about medical devices based on the provided regulatory documents.

IMPORTANT GUIDELINES:
1. Base your answer ONLY on the provided regulatory documents
2. Provide a structured response with clear sections
3. Include specific citations for each major point
4. If the documents don't contain enough information, clearly state this
5. Use professional, technical language appropriate for regulatory context
6. Highlight key requirements, processes, or standards mentioned
7. Compare FDA vs WHO approaches when relevant

RESPONSE STRUCTURE:
- ## Executive Summary (brief overview)
- ## Detailed Analysis (main content with citations)
- ## Key Requirements/Standards (if applicable)
- ## Sources (list all referenced documents)

Use citation format: [Source Name, Page X] after each major point.

QUESTION: {question}

CURRENT ANSWER TO REFORMAT: {answer}
{doc_context}

Please reformat the current answer following the guidelines above."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a regulatory expert specializing in medical device regulations. Format responses according to the provided structure."},
                    {"role": "user", "content": regulatory_prompt.format(
                        question=question,
                        answer=answer,
                        doc_context=doc_context
                    )},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            
            formatted_answer = response.choices[0].message.content.strip()
            logger.info("Successfully formatted answer using LLM regulatory prompt")
            return formatted_answer
            
        except Exception as e:
            logger.warning(f"LLM formatting failed: {e}. Falling back to basic formatting.")
            return self._format_answer_basic(answer, question, docs)

    def _format_answer_basic(self, answer: str, question: str = "", docs: List[Dict] = None) -> str:
        """Basic formatting fallback when LLM is not available."""
        if not answer.strip():
            return "No answer provided."
        
        # Clean up the answer
        formatted = answer.strip()
        
        # Ensure proper capitalization
        if formatted and not formatted[0].isupper():
            formatted = formatted[0].upper() + formatted[1:]
        
        # Ensure proper ending punctuation
        if formatted and formatted[-1] not in '.!?':
            formatted += '.'
        
        # Basic regulatory structure
        structured_answer = f"## Executive Summary\n\n{formatted}\n\n"
        
        # Add sources section
        if docs:
            sources = []
            for doc in docs:
                source = doc.get('source', 'Unknown source')
                if source != 'Unknown source':
                    sources.append(source)
            
            if sources:
                structured_answer += f"## Sources\n\n"
                for source in sorted(set(sources)):
                    structured_answer += f"- {source}\n"
        
        return structured_answer

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from the answer."""
        # Split by sentences and filter meaningful ones
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short sentences, questions, headers, and source citations
            if (len(sentence.split()) >= 4 and 
                not sentence.startswith('#') and
                not sentence.startswith('Sources:') and
                not sentence.startswith('-') and
                not sentence.endswith('?') and
                not re.match(r'^\[.*\]$', sentence)):  # Skip citation-only lines
                claims.append(sentence)
        
        return claims

    def _check_support(self, claims: List[str], docs: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Check which claims are supported by the retrieved documents."""
        doc_texts = []
        for doc in docs:
            content = doc.get("content", "")
            if isinstance(content, str):
                doc_texts.append(content.lower())
        
        supported, unsupported = [], []
        
        for claim in claims:
            # Extract key terms from the claim, excluding common regulatory words
            tokens = [t for t in re.findall(r'\w+', claim.lower()) 
                     if len(t) > 3 and t not in {'medical', 'device', 'regulation', 'requirement', 'guidance'}][:7]
            
            if not tokens:
                unsupported.append(claim)
                continue
            
            # Check if sufficient tokens appear in documents
            support_count = 0
            for token in tokens:
                if any(token in doc_text for doc_text in doc_texts):
                    support_count += 1
            
            # Require at least 30% of key terms to be found
            if support_count >= max(1, len(tokens) * 0.3):
                supported.append(claim)
            else:
                unsupported.append(claim)
        
        return supported, unsupported

    def _make_decision(self, coverage: float, unsupported: List[str]) -> str:
        """Make approval decision based on coverage and unsupported claims."""
        if coverage >= self.min_coverage and len(unsupported) <= self.max_unsupported:
            return "approve"
        return "revise"

    def _get_llm_assessment(
        self,
        question: Optional[str],
        answer: str,
        claims: List[str],
        supported: List[str],
        docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get LLM assessment of regulatory answer quality."""
        try:
            sources = [d.get('source', 'Unknown') for d in docs[:3]]
            
            prompt = f"""Evaluate this regulatory answer for quality and accuracy. Return JSON only.

Question: {question}
Answer: {answer}
Supported claims: {len(supported)}/{len(claims)}
Available sources: {sources}

Return JSON with:
- overall_quality: 1-5 scale
- regulatory_compliance: 1-5 scale for regulatory formatting
- citation_quality: 1-5 scale for proper citations
- missing_aspects: list of missing important regulatory aspects
- potential_hallucinations: list of potentially unsupported statements
- formatting_score: 1-5 for clarity and regulatory structure"""

            response = self.client.chat.completions.create(
                model=self.validation_model,
                messages=[
                    {"role": "system", "content": "You are a regulatory quality assessor. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"raw_response": raw_content}
                
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}")
            return {"error": str(e)}

    def _error_result(self, reason: str) -> Dict[str, Any]:
        """Return error result structure."""
        return {
            "decision": "revise",
            "formatted_answer": "",
            "reason": reason,
            "metrics": {"coverage": 0.0, "total_claims": 0, "unsupported": 0},
            "unsupported_claims": [],
            "llm_assessment": {},
        }