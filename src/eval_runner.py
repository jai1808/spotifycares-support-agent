import json
import logging
from typing import Any, Dict, List
import pandas as pd
from datetime import datetime

from .evaluator import intent_metrics, escalation_metrics, run_llm_judge

logger = logging.getLogger(__name__)

def load_golden_eval(path: str) -> list[dict]:
    """Load golden evaluation set from JSON."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("examples", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.error(f"Failed to load golden eval set from {path}: {e}")
        return []

def run_evaluation(agent: Any, eval_set: list[dict], judge: bool = True, max_examples: int | None = None, judge_every: int = 1) -> dict:
    """Run a full evaluation of an agent on the golden eval set.
    
    For each example:
    1. Call agent.process(user_tweet)
    2. Collect intent prediction, escalation decision, draft reply
    3. Compute intent and escalation metrics
    4. Optionally run LLM judge on each reply (or every judge_every-th example)
    
    Returns comprehensive results dict with:
    - intent_metrics
    - escalation_metrics  
    - judge_scores (if judge=True): mean scores across all examples
    - per_example_results: list of individual results
    
    Print progress as it goes.
    """
    if max_examples is not None:
        eval_set = eval_set[:max_examples]

    intent_preds = []
    intent_truths = []
    esc_preds = []
    esc_truths = []
    per_example_results = []
    judge_scores_accum = {"empathy": [], "helpfulness": [], "safety": [], "overall": []}
    
    print(f"Starting evaluation on {len(eval_set)} examples (judge={judge}, judge_every={judge_every})...")
    logger.info(f"Starting evaluation on {len(eval_set)} examples...")
    
    for i, item in enumerate(eval_set):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(eval_set)}...")
            
        user_tweet = item.get("user_tweet", "")
        ground_truth_intent = item.get("intent", "unknown")
        ground_truth_escalate = item.get("should_escalate", False)
        reference_reply = item.get("brand_reply", "")
        
        # Extract from auto_labels if present
        if "auto_labels" in item and item["auto_labels"]:
            ground_truth_intent = item["auto_labels"].get("intent", ground_truth_intent)
            ground_truth_escalate = item["auto_labels"].get("should_escalate", ground_truth_escalate)
            
        try:
            agent_result = agent.process(user_tweet)
            pred_intent = agent_result.get("intent", "unknown")
            pred_escalate = agent_result.get("should_escalate", False)
            draft_reply = agent_result.get("draft_reply", "")
        except Exception as e:
            logger.error(f"Agent failed on example {i}: {e}")
            pred_intent = "error"
            pred_escalate = False
            draft_reply = ""
            
        intent_preds.append(pred_intent)
        intent_truths.append(ground_truth_intent)
        esc_preds.append(pred_escalate)
        esc_truths.append(ground_truth_escalate)
        
        example_res = {
            "tweet": user_tweet,
            "truth_intent": ground_truth_intent,
            "pred_intent": pred_intent,
            "truth_escalate": ground_truth_escalate,
            "pred_escalate": pred_escalate,
            "reference_reply": reference_reply,
            "draft_reply": draft_reply,
            "judge_result": None
        }
        
        do_judge = judge and draft_reply and reference_reply and (i % max(judge_every, 1) == 0)
        if do_judge:
            judge_res = run_llm_judge(user_tweet, draft_reply, reference_reply, pred_intent)
            example_res["judge_result"] = judge_res
            judge_scores_accum["empathy"].append(judge_res.get("empathy_score", 0))
            judge_scores_accum["helpfulness"].append(judge_res.get("helpfulness_score", 0))
            judge_scores_accum["safety"].append(judge_res.get("safety_score", 0))
            judge_scores_accum["overall"].append(judge_res.get("overall_score", 0.0))
            
        per_example_results.append(example_res)
        
    print("Computing final metrics...")
    i_metrics = intent_metrics(intent_preds, intent_truths)
    e_metrics = escalation_metrics(esc_preds, esc_truths)
    
    final_results = {
        "intent_metrics": i_metrics,
        "escalation_metrics": e_metrics,
        "per_example_results": per_example_results,
    }
    
    if judge and judge_scores_accum["overall"]:
        n = len(judge_scores_accum["overall"])
        final_results["judge_scores"] = {
            "mean_empathy": sum(judge_scores_accum["empathy"]) / n,
            "mean_helpfulness": sum(judge_scores_accum["helpfulness"]) / n,
            "mean_safety": sum(judge_scores_accum["safety"]) / n,
            "mean_overall": sum(judge_scores_accum["overall"]) / n,
            "n_judged": n,
        }
        
    return final_results

def generate_report(results: dict, agent_name: str, output_path: str):
    """Generate a markdown evaluation report from results."""
    report = f"# Evaluation Report: {agent_name}\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Intent Classification Metrics\n"
    i_mets = results.get("intent_metrics", {})
    report += f"- Accuracy: {i_mets.get('accuracy', 0):.4f}\n"
    report += f"- F1 Macro: {i_mets.get('f1_macro', 0):.4f}\n"
    report += f"- F1 Weighted: {i_mets.get('f1_weighted', 0):.4f}\n\n"
    
    report += "## Escalation Metrics\n"
    e_mets = results.get("escalation_metrics", {})
    report += f"- Accuracy: {e_mets.get('accuracy', 0):.4f}\n"
    report += f"- Precision: {e_mets.get('precision', 0):.4f}\n"
    report += f"- Recall: {e_mets.get('recall', 0):.4f}\n"
    report += f"- F1 Score: {e_mets.get('f1', 0):.4f}\n\n"
    
    if "judge_scores" in results:
        report += "## LLM Judge Scores\n"
        j_mets = results["judge_scores"]
        report += f"- Mean Empathy: {j_mets.get('mean_empathy', 0):.2f}/5\n"
        report += f"- Mean Helpfulness: {j_mets.get('mean_helpfulness', 0):.2f}/5\n"
        report += f"- Mean Safety: {j_mets.get('mean_safety', 0):.2f}/5\n"
        report += f"- Mean Overall: {j_mets.get('mean_overall', 0):.2f}/5\n\n"
        
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

def compare_systems(results_dict: dict[str, dict], output_path: str):
    """Compare multiple systems and generate comparison table.
    results_dict maps system_name -> results from run_evaluation.
    """
    records = []
    for sys_name, res in results_dict.items():
        i_mets = res.get("intent_metrics", {})
        e_mets = res.get("escalation_metrics", {})
        j_mets = res.get("judge_scores", {})
        
        records.append({
            "System": sys_name,
            "Intent Acc": i_mets.get("accuracy", 0),
            "Intent F1 (W)": i_mets.get("f1_weighted", 0),
            "Escalation F1": e_mets.get("f1", 0),
            "Judge Overall": j_mets.get("mean_overall", 0)
        })
        
    df = pd.DataFrame(records)
    md_table = df.to_markdown(index=False)
    
    report = "# System Comparison\n\n" + md_table + "\n"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Comparison report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save comparison report: {e}")
