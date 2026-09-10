"""
Interactive demo for the SpotifyCares AI Support Agent.

Lets you type in tweets and see the agent's classification + reply in real-time.
Useful for quick testing and demonstrations.

Usage:
    python scripts/demo.py
    python scripts/demo.py --agent zero-shot  # Use zero-shot instead of RAG
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="SpotifyCares Agent Demo")
    parser.add_argument(
        "--agent",
        choices=["trivial", "zero-shot", "rag"],
        default="rag",
        help="Which agent to demo (default: rag)",
    )
    args = parser.parse_args()

    # Initialize agent
    if args.agent == "trivial":
        from src.agent import TrivialBaseline
        agent = TrivialBaseline()
        print("Using: Trivial Baseline")
    elif args.agent == "zero-shot":
        from src.agent import ZeroShotAgent
        agent = ZeroShotAgent()
        print("Using: Zero-Shot Agent")
    else:
        from src.agent import RAGAgent
        from src.rag import RAGRetriever
        index_path = str(DATA_DIR / "rag_index")
        try:
            retriever = RAGRetriever.load(index_path)
            agent = RAGAgent(retriever)
            print("Using: RAG Agent (loaded index)")
        except FileNotFoundError:
            print("RAG index not found. Falling back to Zero-Shot Agent.")
            print("Run 'python run_pipeline.py --step rag' to build the index.")
            from src.agent import ZeroShotAgent
            agent = ZeroShotAgent()

    print("\n" + "=" * 60)
    print("  SpotifyCares AI Support Agent — Interactive Demo")
    print("=" * 60)
    print("\nType a customer tweet and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            tweet = input("🎵 Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not tweet or tweet.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.process(tweet)
        print(f"\n  📋 Intent: {result.get('intent', 'Unknown')}")
        print(f"  🚨 Escalate: {result.get('should_escalate', False)}")
        if result.get("escalation_reason"):
            print(f"  📝 Reason: {result['escalation_reason']}")
        print(f"  💬 Reply: {result.get('draft_reply', '')}")
        print()


if __name__ == "__main__":
    main()
