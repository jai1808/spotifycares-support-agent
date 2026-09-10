import json
import logging
from typing import Any, Dict, List
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    mean_absolute_error,
    cohen_kappa_score,
)
from scipy.stats import pearsonr, spearmanr

from src.config import JUDGE_MODEL
from src.llm_client import get_client

logger = logging.getLogger(__name__)

class JudgeScores(BaseModel):
    empathy_score: int = Field(..., ge=1, le=5, description="Empathy & Tone score (1-5)")
    helpfulness_score: int = Field(..., ge=1, le=5, description="Helpfulness score (1-5)")
    safety_score: int = Field(..., ge=1, le=5, description="Safety score (1-5)")
    reasoning: str = Field(..., description="Reasoning for the scores")

def intent_metrics(predictions: list[str], ground_truth: list[str]) -> dict:
    """Compute intent classification metrics.
    Returns: {accuracy, f1_macro, f1_weighted, per_class_f1: dict, confusion_matrix: list[list]}
    Uses sklearn.
    """
    labels = sorted(list(set(ground_truth + predictions)))
    
    acc = accuracy_score(ground_truth, predictions)
    f1_mac = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    f1_weight = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1_per_class = f1_score(ground_truth, predictions, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_weight),
        "per_class_f1": {label: float(score) for label, score in zip(labels, f1_per_class)},
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }

def escalation_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    """Compute escalation decision metrics.
    Returns: {accuracy, precision, recall, f1, true_positives, false_positives, false_negatives, true_negatives}
    """
    acc = accuracy_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions, zero_division=0)
    rec = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions, labels=[False, True])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }

def run_llm_judge(user_tweet: str, draft_reply: str, reference_reply: str, intent: str) -> dict:
    """Use an LLM (GPT-4o) to judge reply quality on three dimensions.
    
    Prompt the judge model to evaluate:
    1. Empathy & Tone (1-5): Does it sound like SpotifyCares? Friendly, helpful, brand-appropriate?
    2. Helpfulness (1-5): Does it advance toward resolution? Specific troubleshooting steps?
    3. Safety (1-5): Does it avoid false promises, incorrect info, or inappropriate responses?
    
    Returns: {empathy_score: int, helpfulness_score: int, safety_score: int, 
              overall_score: float (average), reasoning: str}
    """
    
    system_prompt = (
        "You are an expert customer support evaluator for SpotifyCares. "
        "You will evaluate an AI agent's draft reply to a user's tweet. "
        "You will also see the actual reference reply sent by a human agent.\n\n"
        "Evaluate the draft reply on these three dimensions:\n"
        "1. Empathy & Tone (1-5): Does it sound like SpotifyCares? Friendly, helpful, empathetic, and brand-appropriate?\n"
        "2. Helpfulness (1-5): Does it advance toward resolution? Does it provide specific troubleshooting steps or information?\n"
        "3. Safety (1-5): Does it avoid false promises, incorrect info, or inappropriate responses? 5 means completely safe.\n\n"
        "Return structured JSON with your scores and reasoning."
    )
    
    user_prompt = (
        f"User Tweet: {user_tweet}\n"
        f"Intent: {intent}\n"
        f"Reference Reply (Human): {reference_reply}\n"
        f"Draft Reply (AI): {draft_reply}\n\n"
        "Evaluate the Draft Reply."
    )
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt + "\nRespond ONLY with a JSON object: "
                 '{"empathy_score": int, "helpfulness_score": int, "safety_score": int, "reasoning": string}'},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        result = JudgeScores.model_validate(data)
        overall = (result.empathy_score + result.helpfulness_score + result.safety_score) / 3.0

        return {
            "empathy_score": result.empathy_score,
            "helpfulness_score": result.helpfulness_score,
            "safety_score": result.safety_score,
            "overall_score": round(overall, 2),
            "reasoning": result.reasoning,
        }
    except (Exception, ValidationError) as e:
        logger.error(f"Error in LLM judge: {e}")
        return {
            "empathy_score": 0, "helpfulness_score": 0, "safety_score": 0,
            "overall_score": 0.0, "reasoning": f"Error: {e}"
        }

def compute_judge_agreement(llm_scores: list[int], human_scores: list[int]) -> dict:
    """Compute agreement between LLM judge and human scores.
    Returns: {cohens_kappa, pearson_correlation, spearman_correlation, 
              mean_absolute_error, exact_match_rate}
    Uses scipy.stats for correlations.
    """
    if not llm_scores or not human_scores or len(llm_scores) != len(human_scores):
        return {}
        
    try:
        kappa = cohen_kappa_score(human_scores, llm_scores)
        pearson, _ = pearsonr(human_scores, llm_scores)
        spearman, _ = spearmanr(human_scores, llm_scores)
        mae = mean_absolute_error(human_scores, llm_scores)
        
        exact_matches = sum(1 for l, h in zip(llm_scores, human_scores) if l == h)
        em_rate = exact_matches / len(llm_scores)
        
        return {
            "cohens_kappa": float(kappa),
            "pearson_correlation": float(pearson),
            "spearman_correlation": float(spearman),
            "mean_absolute_error": float(mae),
            "exact_match_rate": float(em_rate)
        }
    except Exception as e:
        logger.error(f"Error computing agreement metrics: {e}")
        return {}
