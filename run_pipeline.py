"""
Main pipeline script for the SpotifyCares Customer Support AI Agent.

This script orchestrates the full pipeline:
1. Download / load dataset
2. Extract SpotifyCares conversations
3. Build golden evaluation set (200 hand-labeled examples)
4. Build RAG corpus + index
5. Run all 3 systems (Trivial, Zero-Shot, RAG) on the golden eval set
6. Compute metrics + LLM judge scores
7. Generate comparative report

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --step data        # Only data prep
    python run_pipeline.py --step golden      # Only golden set
    python run_pipeline.py --step eval        # Only evaluation (requires golden set)
    python run_pipeline.py --step eval --no-judge  # Skip LLM judge (faster)
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.config import (
    DATA_DIR, EVAL_DIR, OUTPUTS_DIR, RAW_CSV,
    RAG_CORPUS_PATH, GOLDEN_EVAL_PATH, TARGET_BRAND,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def step_data():
    """Step 1: Load data and extract brand conversations."""
    from src.data_loader import load_raw_data, get_brand_conversations, get_first_contact_pairs

    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 60)

    if not RAW_CSV.exists():
        print(f"\nWARNING: Dataset not found at {RAW_CSV}")
        print("  Please download it first:")
        print("    python scripts/download_data.py")
        print("  Or manually download from:")
        print("    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        sys.exit(1)

    # Load raw data
    df = load_raw_data(str(RAW_CSV))
    print(f"\nDataset shape: {df.shape}")
    print(f"Unique authors: {df['author_id'].nunique()}")

    # Show top brands
    brand_counts = df[df['inbound'] == False]['author_id'].value_counts().head(20)
    print(f"\nTop 20 brands by tweet count:")
    for brand, count in brand_counts.items():
        marker = " ← TARGET" if brand == TARGET_BRAND else ""
        print(f"  {brand}: {count}{marker}")

    # Extract brand conversations
    brand_convos = get_brand_conversations(df, TARGET_BRAND)
    print(f"\n{TARGET_BRAND} conversation pairs: {len(brand_convos)}")

    # Get first contact pairs
    first_contacts = get_first_contact_pairs(brand_convos)
    print(f"First-contact pairs: {len(first_contacts)}")

    # Save intermediate data
    pairs_path = DATA_DIR / "first_contact_pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(first_contacts, f, indent=2, ensure_ascii=False)
    print(f"\nSaved first-contact pairs to {pairs_path}")

    # Save brand conversations for RAG
    brand_convos_path = DATA_DIR / "brand_conversations.json"
    brand_convos.to_json(brand_convos_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved brand conversations to {brand_convos_path}")

    return first_contacts, brand_convos


def step_golden(first_contacts=None):
    """Step 2: Build the golden evaluation set."""
    from src.golden_set_builder import build_golden_set

    print("\n" + "=" * 60)
    print("STEP 2: BUILDING GOLDEN EVALUATION SET")
    print("=" * 60)

    if first_contacts is None:
        pairs_path = DATA_DIR / "first_contact_pairs.json"
        if not pairs_path.exists():
            print(f"First-contact pairs not found at {pairs_path}. Run --step data first.")
            sys.exit(1)
        with open(pairs_path, "r", encoding="utf-8") as f:
            first_contacts = json.load(f)

    print(f"\nBuilding golden set from {len(first_contacts)} first-contact pairs...")
    labeled = build_golden_set(
        first_contacts,
        str(GOLDEN_EVAL_PATH),
        n_samples=200,
    )
    return labeled


def step_rag(brand_convos=None):
    """Step 3: Build RAG corpus and index."""
    import pandas as pd
    from src.rag import build_rag_corpus, save_rag_corpus, RAGRetriever

    print("\n" + "=" * 60)
    print("STEP 3: BUILDING RAG CORPUS & INDEX")
    print("=" * 60)

    if brand_convos is None:
        convos_path = DATA_DIR / "brand_conversations.json"
        if not convos_path.exists():
            print(f"Brand conversations not found. Run --step data first.")
            sys.exit(1)
        brand_convos = pd.read_json(convos_path)

    # Build corpus (limit to 5000 for embedding cost management)
    corpus = build_rag_corpus(brand_convos)
    print(f"\nTotal corpus size: {len(corpus)}")

    # Sample if too large (keep costs reasonable)
    MAX_CORPUS = 5000
    if len(corpus) > MAX_CORPUS:
        import random
        random.seed(42)
        corpus = random.sample(corpus, MAX_CORPUS)
        print(f"Sampled down to {MAX_CORPUS} for embedding")

    # Save corpus
    save_rag_corpus(corpus, str(RAG_CORPUS_PATH))
    print(f"Saved RAG corpus to {RAG_CORPUS_PATH}")

    # Build embedding index
    print("\nBuilding embedding index (this may take a few minutes)...")
    retriever = RAGRetriever(corpus)
    retriever.build_index()

    index_path = str(DATA_DIR / "rag_index")
    retriever.save(index_path)
    print(f"Saved RAG index to {index_path}")

    return retriever


def step_eval(retriever=None, use_judge: bool = True, max_examples: int = None, judge_every: int = 5):
    """Step 4: Run evaluation on all systems."""
    from src.agent import TrivialBaseline, ZeroShotAgent, RAGAgent
    from src.eval_runner import load_golden_eval, run_evaluation, generate_report, compare_systems
    from src.rag import RAGRetriever

    print("\n" + "=" * 60)
    print("STEP 4: RUNNING EVALUATION")
    print("=" * 60)

    # Load golden eval set
    eval_set = load_golden_eval(str(GOLDEN_EVAL_PATH))
    if not eval_set:
        print("Golden eval set is empty. Run --step golden first.")
        sys.exit(1)
    print(f"\nLoaded {len(eval_set)} evaluation examples.")

    # Load retriever if not provided
    if retriever is None:
        index_path = str(DATA_DIR / "rag_index")
        try:
            retriever = RAGRetriever.load(index_path)
            print(f"Loaded RAG index from {index_path}")
        except Exception as e:
            print(f"Could not load RAG index: {e}")
            print("Run --step rag first, or running without RAG agent.")
            retriever = None

    # Initialize agents
    agents = {
        "Trivial_Baseline": TrivialBaseline(),
        "ZeroShot_Agent": ZeroShotAgent(),
    }
    if retriever is not None:
        agents["RAG_Agent"] = RAGAgent(retriever)

    # Run evaluation for each agent
    all_results = {}
    for agent_name, agent in agents.items():
        print(f"\n{'-'*40}")
        print(f"Evaluating: {agent_name}")
        print(f"{'-'*40}")

        t0 = time.time()
        results = run_evaluation(
            agent,
            eval_set,
            judge=use_judge,
            max_examples=max_examples,
            judge_every=judge_every if use_judge else 1,
        )
        elapsed = time.time() - t0

        results["elapsed_seconds"] = elapsed
        all_results[agent_name] = results

        # Print summary
        i_met = results.get("intent_metrics", {})
        e_met = results.get("escalation_metrics", {})
        j_met = results.get("judge_scores", {})

        print(f"\n  Intent F1 (macro): {i_met.get('f1_macro', 0):.3f}")
        print(f"  Intent F1 (weighted): {i_met.get('f1_weighted', 0):.3f}")
        print(f"  Escalation Precision: {e_met.get('precision', 0):.3f}")
        print(f"  Escalation Recall: {e_met.get('recall', 0):.3f}")
        print(f"  Escalation F1: {e_met.get('f1', 0):.3f}")
        if j_met:
            print(f"  Judge Overall: {j_met.get('mean_overall', 0):.2f}/5 (n={j_met.get('n_judged', '?')})")
        print(f"  Time: {elapsed:.1f}s")

        # Save individual report
        report_path = str(OUTPUTS_DIR / f"report_{agent_name}.md")
        generate_report(results, agent_name, report_path)

        # Save raw results
        results_path = str(OUTPUTS_DIR / f"results_{agent_name}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Generate comparison
    comparison_path = str(OUTPUTS_DIR / "comparison.md")
    compare_systems(all_results, comparison_path)

    # Save all results
    with open(str(OUTPUTS_DIR / "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Compact metrics for README
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "intent_metrics": res.get("intent_metrics", {}),
            "escalation_metrics": res.get("escalation_metrics", {}),
            "judge_scores": res.get("judge_scores", {}),
            "elapsed_seconds": res.get("elapsed_seconds"),
        }
    with open(str(OUTPUTS_DIR / "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nReports saved to: {OUTPUTS_DIR}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="SpotifyCares AI Support Agent Pipeline")
    parser.add_argument(
        "--step",
        choices=["data", "golden", "rag", "eval", "all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge scoring (faster evaluation)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit eval set size (default: all 200)",
    )
    parser.add_argument(
        "--judge-every",
        type=int,
        default=5,
        help="Run LLM judge every N examples (default: 5) to save time/cost",
    )
    args = parser.parse_args()

    # Ensure output dirs exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("  SpotifyCares Customer Support AI Agent Pipeline")
    print("=======================================================\n")

    if args.step == "data":
        step_data()
    elif args.step == "golden":
        step_golden()
    elif args.step == "rag":
        step_rag()
    elif args.step == "eval":
        step_eval(
            use_judge=not args.no_judge,
            max_examples=args.max_examples,
            judge_every=args.judge_every,
        )
    elif args.step == "all":
        first_contacts, brand_convos = step_data()
        step_golden(first_contacts)
        retriever = step_rag(brand_convos)
        step_eval(
            retriever,
            use_judge=not args.no_judge,
            max_examples=args.max_examples,
            judge_every=args.judge_every,
        )


if __name__ == "__main__":
    main()
