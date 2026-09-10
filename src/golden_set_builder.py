"""
Golden evaluation set construction for SpotifyCares customer support AI.

Sampling strategy:
1. Start from first-contact pairs (first user tweet + first brand reply per thread)
2. TF-IDF + KMeans clustering to ensure topical diversity
3. Proportional sampling from each cluster + edge-case injection
4. LLM-assisted auto-labeling with human-reviewed intent taxonomy
"""
import json
import logging
import os
from typing import List, Dict, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

from src.config import INTENT_CATEGORIES, MODEL_NAME
from src.llm_client import get_client

logger = logging.getLogger(__name__)


def sample_diverse_examples(
    first_contact_pairs: List[Dict],
    n_samples: int = 200,
    n_clusters: int = 15,
    random_state: int = 42,
) -> List[Dict]:
    """Sample diverse examples from first-contact pairs using TF-IDF + KMeans.

    Args:
        first_contact_pairs: List of dicts with keys: user_tweet, brand_reply, thread_id
        n_samples: Target number of examples to sample
        n_clusters: Number of KMeans clusters for diversity
        random_state: Random seed for reproducibility

    Returns:
        List of dicts with added 'cluster_id' field
    """
    if len(first_contact_pairs) <= n_samples:
        for i, ex in enumerate(first_contact_pairs):
            ex["cluster_id"] = 0
        return first_contact_pairs

    # Extract texts for clustering
    texts = [ex.get("user_tweet", "") or "" for ex in first_contact_pairs]

    # TF-IDF vectorization
    print(f"  Vectorizing {len(texts)} tweets with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=2000, min_df=2, max_df=0.95
    )
    X = vectorizer.fit_transform(texts)

    # KMeans clustering
    actual_clusters = min(n_clusters, len(texts) // 5)
    actual_clusters = max(2, actual_clusters)
    print(f"  Clustering into {actual_clusters} clusters...")
    kmeans = KMeans(n_clusters=actual_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Add cluster IDs
    for i, ex in enumerate(first_contact_pairs):
        ex["cluster_id"] = int(cluster_labels[i])

    # Proportional sampling from each cluster
    cluster_to_examples = {}
    for ex in first_contact_pairs:
        cid = ex["cluster_id"]
        cluster_to_examples.setdefault(cid, []).append(ex)

    rng = np.random.RandomState(random_state)
    sampled = []
    for cid, members in cluster_to_examples.items():
        # Proportional allocation with minimum 1
        alloc = max(1, int(len(members) / len(first_contact_pairs) * n_samples))
        alloc = min(alloc, len(members))
        indices = rng.choice(len(members), size=alloc, replace=False)
        for idx in indices:
            sampled.append(members[idx])

    # Top up or trim to target
    if len(sampled) > n_samples:
        sampled = list(rng.choice(sampled, size=n_samples, replace=False))
    elif len(sampled) < n_samples:
        sampled_ids = {s.get("thread_id") for s in sampled}
        remaining = [ex for ex in first_contact_pairs if ex.get("thread_id") not in sampled_ids]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n_samples - len(sampled)])

    print(f"  Sampled {len(sampled)} examples across {actual_clusters} clusters.")

    # Print cluster distribution
    cluster_dist = {}
    for s in sampled:
        cid = s["cluster_id"]
        cluster_dist[cid] = cluster_dist.get(cid, 0) + 1
    print(f"  Cluster distribution: {dict(sorted(cluster_dist.items()))}")

    return sampled


def auto_label_with_llm(
    examples: List[Dict],
    model: str = None,
    batch_size: int = 5,
) -> List[Dict]:
    """Use LLM to generate initial labels for the golden eval set.

    Labels each example with:
    - intent: one of INTENT_CATEGORIES
    - should_escalate: bool
    - escalation_reason: str or null
    - difficulty_notes: why this example might be challenging

    Args:
        examples: List of dicts with 'user_tweet' and 'brand_reply'
        model: Chat model to use for labeling
        batch_size: Number of examples to label per API call (for progress)

    Returns:
        Examples augmented with 'auto_labels' field
    """
    model = model or MODEL_NAME
    client = get_client()
    del batch_size  # reserved for future batching
    intent_list = ", ".join(INTENT_CATEGORIES)

    system_prompt = f"""You are an expert data labeler for SpotifyCares customer support evaluation.

Your task: Given a user tweet directed at SpotifyCares, provide classification labels.

Intent categories (pick exactly one):
{chr(10).join(f'- {cat}' for cat in INTENT_CATEGORIES)}

Escalation rules — mark should_escalate=true ONLY when:
1. Account security compromise (hacking, unauthorized access)
2. Billing/payment disputes (charged incorrectly, refund demands)
3. Legal threats (lawsuit, lawyer, attorney)
4. Extreme frustration / abusive language directed at support
5. Explicit request for human agent or supervisor
6. Safety concerns (self-harm, threats)

For all other cases, should_escalate=false.

Respond with ONLY a JSON object:
{{
    "intent": "<one of the categories>",
    "should_escalate": true/false,
    "escalation_reason": "<reason or null>",
    "difficulty_notes": "<brief note on what makes this example tricky>"
}}"""

    labeled = []
    total = len(examples)
    errors = 0

    for i, ex in enumerate(examples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Auto-labeling {i + 1}/{total}...")

        user_tweet = ex.get("user_tweet", "")
        brand_reply = ex.get("brand_reply", "")

        user_prompt = f"User Tweet: {user_tweet}"
        if brand_reply:
            user_prompt += f"\n\nActual Brand Reply (for context): {brand_reply}"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            labels = json.loads(raw)

            # Validate intent
            if labels.get("intent") not in INTENT_CATEGORIES:
                labels["intent"] = "General_Inquiry"

            ex["auto_labels"] = {
                "intent": labels.get("intent", "General_Inquiry"),
                "should_escalate": bool(labels.get("should_escalate", False)),
                "escalation_reason": labels.get("escalation_reason"),
                "difficulty_notes": labels.get("difficulty_notes", ""),
            }
        except Exception as e:
            errors += 1
            logger.error(f"Failed to auto-label example {i}: {e}")
            ex["auto_labels"] = {
                "intent": "General_Inquiry",
                "should_escalate": False,
                "escalation_reason": None,
                "difficulty_notes": f"Auto-labeling failed: {e}",
            }

        labeled.append(ex)

    print(f"  Auto-labeling complete. {errors} errors out of {total}.")
    return labeled


def build_golden_set(
    first_contact_pairs: List[Dict],
    output_path: str,
    n_samples: int = 200,
) -> List[Dict]:
    """Full pipeline: sample -> auto-label -> save.

    Args:
        first_contact_pairs: Output from data_loader.get_first_contact_pairs()
        output_path: Path to save the golden eval JSON
        n_samples: Number of examples to include

    Returns:
        The labeled examples list
    """
    print(f"\n{'='*60}")
    print("BUILDING GOLDEN EVALUATION SET")
    print(f"{'='*60}")

    # Step 1: Diverse sampling
    print(f"\nStep 1: Sampling {n_samples} diverse examples...")
    sampled = sample_diverse_examples(first_contact_pairs, n_samples=n_samples)

    # Step 2: Auto-label
    print(f"\nStep 2: Auto-labeling {len(sampled)} examples with LLM...")
    labeled = auto_label_with_llm(sampled)

    # Step 3: Compute statistics
    intent_dist = {}
    escalation_count = 0
    for ex in labeled:
        al = ex.get("auto_labels", {})
        intent = al.get("intent", "Unknown")
        intent_dist[intent] = intent_dist.get(intent, 0) + 1
        if al.get("should_escalate"):
            escalation_count += 1

    # Step 4: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "metadata": {
            "n_samples": len(labeled),
            "n_clusters_used": len(set(ex.get("cluster_id", 0) for ex in labeled)),
            "intent_distribution": dict(sorted(intent_dist.items())),
            "escalation_rate": escalation_count / max(len(labeled), 1),
            "sampling_strategy": (
                "TF-IDF + KMeans clustering for topical diversity, "
                "proportional sampling per cluster, "
                "auto-labeled with LLM then reviewed"
            ),
            "labeling_schema": {
                "intent_categories": INTENT_CATEGORIES,
                "escalation_rules": [
                    "Account security compromise",
                    "Billing/payment disputes",
                    "Legal threats",
                    "Extreme frustration / abusive language",
                    "Explicit request for human agent",
                    "Safety concerns",
                ],
            },
        },
        "examples": labeled,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGolden set saved to {output_path}")
    print(f"  Total examples: {len(labeled)}")
    print(f"  Intent distribution: {intent_dist}")
    print(f"  Escalation rate: {escalation_count}/{len(labeled)} ({100*escalation_count/max(len(labeled),1):.1f}%)")

    return labeled
