"""Project configuration constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_CSV = DATA_DIR / "twcs.csv"
RAG_CORPUS_PATH = DATA_DIR / "rag_corpus.json"
GOLDEN_EVAL_PATH = EVAL_DIR / "golden_eval.json"

# Brand
TARGET_BRAND = "SpotifyCares"

# Intent taxonomy
INTENT_CATEGORIES = [
    "Account_Access",
    "Billing_Payment",
    "Playback_Issue",
    "App_Bug_Crash",
    "Content_Availability",
    "Subscription_Plan",
    "Feature_Request",
    "General_Inquiry",
]

# Escalation triggers
ESCALATION_KEYWORDS = [
    "lawyer", "lawsuit", "sue", "legal", "attorney",
    "hack", "hacked", "stolen", "unauthorized",
    "cancel", "refund", "charged", "billing",
    "kill", "die", "threat",
]

# LLM (DeepSeek via OpenAI-compatible API; override in .env)
MODEL_NAME = os.environ.get("OPENAI_MODEL", "deepseek-chat")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-chat")
# Local TF-IDF retrieval (no cloud embedding API required)
EMBEDDING_MODEL = "tfidf-local"
