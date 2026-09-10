import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.llm_client import get_client
from src.rag import RAGRetriever
from src.config import MODEL_NAME, INTENT_CATEGORIES


class BaseAgent(ABC):
    @abstractmethod
    def process(self, user_tweet: str) -> Dict[str, Any]:
        """
        Process a user tweet and return intent classification and drafted reply.
        Returns a dict: {"intent": str, "should_escalate": bool, "escalation_reason": str|None, "draft_reply": str}
        """
        pass

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Helper to parse JSON from LLM response with fallback handling."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
            except Exception:
                pass
                
        return {
            "intent": "Unknown",
            "should_escalate": True,
            "escalation_reason": "Failed to parse LLM response",
            "draft_reply": "I'm sorry, I encountered an error while processing your request. Please try again later."
        }


class TrivialBaseline(BaseAgent):
    def process(self, user_tweet: str) -> Dict[str, Any]:
        return {
            "intent": "General_Inquiry",
            "should_escalate": True,
            "escalation_reason": "Trivial baseline always escalates",
            "draft_reply": ""
        }


class ZeroShotAgent(BaseAgent):
    def __init__(self):
        self.client = get_client()
        
    def process(self, user_tweet: str) -> Dict[str, Any]:
        system_prompt = f"""You are a customer support AI for SpotifyCares.
Your task is to classify the intent of user tweets, decide if the issue should be escalated to a human agent, and draft a helpful reply.

Intent categories:
{', '.join(INTENT_CATEGORIES)}

Respond ONLY with a JSON object in this format:
{{
    "intent": "One of the intent categories",
    "should_escalate": boolean,
    "escalation_reason": "String explaining why, or null if false",
    "draft_reply": "Your drafted reply to the user"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_tweet}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content or "{}"
            return self._parse_json_response(response_text)
        except Exception as e:
            return {
                "intent": "Error",
                "should_escalate": True,
                "escalation_reason": f"API Error: {str(e)}",
                "draft_reply": ""
            }


class RAGAgent(BaseAgent):
    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.client = get_client()
        
    def process(self, user_tweet: str) -> Dict[str, Any]:
        similar_conversations = self.retriever.retrieve(user_tweet, top_k=3)
        
        context_str = ""
        for i, conv in enumerate(similar_conversations, 1):
            context_str += f"--- Example {i} ---\nUser: {conv['user_query']}\nBrand Reply: {conv['brand_response']}\n\n"
            
        system_prompt = f"""You are a customer support AI for SpotifyCares. You are friendly, empathetic, and occasionally use emoji. You offer specific troubleshooting steps when appropriate.

Intent Categories:
{', '.join(INTENT_CATEGORIES)}

Escalation Rules:
Escalate ONLY when:
1. Account security is compromised
2. There are billing disputes
3. User threatens legal action
4. User exhibits extreme frustration
5. User explicitly requests a supervisor or human agent

Here are some historical examples of similar conversations:
{context_str}

Analyze the user's tweet based on the guidelines and historical context. Respond ONLY with a JSON object:
{{
    "intent": "String (one of the intent categories)",
    "should_escalate": boolean,
    "escalation_reason": "String explaining why based on rules, or null",
    "draft_reply": "Your friendly, empathetic drafted reply (in SpotifyCares tone)"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_tweet}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content or "{}"
            return self._parse_json_response(response_text)
        except Exception as e:
            return {
                "intent": "Error",
                "should_escalate": True,
                "escalation_reason": f"API Error: {str(e)}",
                "draft_reply": ""
            }
