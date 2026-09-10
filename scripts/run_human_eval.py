"""
Human-Judge Agreement Evaluation Script.

This script:
1. Loads a subset of evaluation results (50 examples)
2. Presents them for human scoring (empathy, helpfulness, safety on 1-5 scale)
3. Computes Cohen's Kappa and correlation between human and LLM judge scores
4. Saves the alignment report

Usage:
    python scripts/run_human_eval.py
    python scripts/run_human_eval.py --input outputs/results_RAG_Agent.json --n 50
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import compute_judge_agreement


def load_judged_results(path: str) -> list[dict]:
    """Load results that have LLM judge scores."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_example = data.get("per_example_results", [])
    judged = [ex for ex in per_example if ex.get("judge_result") and ex["judge_result"].get("overall_score", 0) > 0]
    return judged


def collect_human_scores(examples: list[dict]) -> list[dict]:
    """Present examples to human and collect scores."""
    print("\n" + "=" * 60)
    print("HUMAN EVALUATION")
    print("=" * 60)
    print(f"\nYou will be shown {len(examples)} examples.")
    print("For each, rate the AI's draft reply on a 1-5 scale:")
    print("  1 = Very Poor, 2 = Poor, 3 = Adequate, 4 = Good, 5 = Excellent")
    print("  0 = Skip this example")
    print("\nDimensions:")
    print("  Empathy: Does it sound like SpotifyCares? Friendly, empathetic?")
    print("  Helpfulness: Does it advance toward resolution?")
    print("  Safety: Does it avoid false promises or bad advice?")
    print("=" * 60)

    scored = []
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i + 1}/{len(examples)} ---")
        print(f"User Tweet: {ex['tweet']}")
        print(f"Reference Reply: {ex['reference_reply']}")
        print(f"AI Draft Reply: {ex['draft_reply']}")
        print()

        try:
            empathy = int(input("  Empathy (1-5, 0 to skip): "))
            if empathy == 0:
                continue
            helpfulness = int(input("  Helpfulness (1-5): "))
            safety = int(input("  Safety (1-5): "))

            scored.append({
                "index": i,
                "tweet": ex["tweet"],
                "llm_empathy": ex["judge_result"]["empathy_score"],
                "llm_helpfulness": ex["judge_result"]["helpfulness_score"],
                "llm_safety": ex["judge_result"]["safety_score"],
                "human_empathy": max(1, min(5, empathy)),
                "human_helpfulness": max(1, min(5, helpfulness)),
                "human_safety": max(1, min(5, safety)),
            })
        except (ValueError, EOFError):
            print("  Skipping...")
            continue

    return scored


def compute_alignment(scored: list[dict]) -> dict:
    """Compute alignment metrics between human and LLM scores."""
    results = {}

    for dim in ["empathy", "helpfulness", "safety"]:
        llm_scores = [s[f"llm_{dim}"] for s in scored]
        human_scores = [s[f"human_{dim}"] for s in scored]
        agreement = compute_judge_agreement(llm_scores, human_scores)
        results[dim] = agreement

    # Overall (averaged)
    llm_overall = [
        (s["llm_empathy"] + s["llm_helpfulness"] + s["llm_safety"]) / 3
        for s in scored
    ]
    human_overall = [
        (s["human_empathy"] + s["human_helpfulness"] + s["human_safety"]) / 3
        for s in scored
    ]
    # Round for kappa
    llm_rounded = [round(x) for x in llm_overall]
    human_rounded = [round(x) for x in human_overall]
    results["overall"] = compute_judge_agreement(llm_rounded, human_rounded)

    return results


def main():
    parser = argparse.ArgumentParser(description="Human-Judge Agreement Evaluation")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "outputs" / "results_RAG_Agent.json"))
    parser.add_argument("--n", type=int, default=50, help="Number of examples to evaluate")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "human_agreement.json"))
    args = parser.parse_args()

    # Load
    judged = load_judged_results(args.input)
    if len(judged) < args.n:
        print(f"Only {len(judged)} judged examples available (requested {args.n})")
        args.n = len(judged)

    # Sample
    random.seed(42)
    sample = random.sample(judged, args.n)

    # Collect human scores
    scored = collect_human_scores(sample)
    if len(scored) < 10:
        print(f"\nOnly {len(scored)} examples scored. Need at least 10 for meaningful statistics.")
        return

    # Compute alignment
    alignment = compute_alignment(scored)

    # Print results
    print("\n" + "=" * 60)
    print("HUMAN-JUDGE AGREEMENT RESULTS")
    print("=" * 60)

    for dim, metrics in alignment.items():
        print(f"\n{dim.upper()}:")
        print(f"  Cohen's Kappa: {metrics.get('cohens_kappa', 'N/A'):.3f}")
        print(f"  Pearson r: {metrics.get('pearson_correlation', 'N/A'):.3f}")
        print(f"  Spearman ρ: {metrics.get('spearman_correlation', 'N/A'):.3f}")
        print(f"  MAE: {metrics.get('mean_absolute_error', 'N/A'):.3f}")
        print(f"  Exact Match: {metrics.get('exact_match_rate', 'N/A'):.1%}")

    # Save
    output = {
        "n_scored": len(scored),
        "alignment_metrics": alignment,
        "scored_examples": scored,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
