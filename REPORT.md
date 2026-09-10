# Evaluation Report: SpotifyCares Customer Support AI

## 1. Problem Framing

This project aims to automate customer support triage and initial response drafting for the `SpotifyCares` brand on Twitter.

**Definition of "Good":**
1.  **Accurate Intent Classification:** Correctly triaging incoming user issues into actionable categories.
2.  **Safe Escalation Decisions:** Reliably identifying when an issue requires a human agent (e.g., account security, legal threats, billing disputes). Recall is far more critical than precision here.
3.  **Brand-Aligned Reply Generation:** Drafting responses that match the distinct, empathetic, and slightly informal tone of historical `SpotifyCares` tweets, while providing specific troubleshooting steps when appropriate.

**What We Chose NOT to Build (and Why):**
*   **Multi-turn conversation handling:** We evaluate only the first user-agent interaction. Managing conversation state introduces complexities that dilute the evaluation of core classification and generation capabilities.
*   **Real-time action execution:** We simulate action (like checking an account) via structured JSON outputs rather than integrating with mock APIs.
*   **Fine-tuned model:** We opted for a Retrieval-Augmented Generation (RAG) architecture. Support policies and features change rapidly; updating a RAG corpus is vastly easier and more maintainable than continuous fine-tuning.

**Why SpotifyCares?**
We selected `SpotifyCares` from the dataset because it presents a balanced challenge:
*   Diverse intent distribution (billing, offline sync, app crashes, content availability).
*   Distinctive brand voice (friendly, frequent use of emojis, specific but concise troubleshooting).
*   Manageable volume within the dataset for rapid prototyping and local TF-IDF indexing cost management.

---

## 2. System Design

The system implements a three-tier analysis of incoming tweets:
1.  **Intent Classification:** Categorizes the tweet into one of 8 distinct intents.
2.  **Escalation (Binary):** Determines if human intervention is required based on strict safety rules.
3.  **Reply Generation:** Drafts a response in the brand's tone.

**Architecture: Retrieval-Augmented Generation (RAG)**
Instead of relying solely on the LLM's parametric knowledge, the system indexes historical `SpotifyCares` resolutions with **local TF-IDF** vectors (numpy cosine similarity; no cloud embedding API). When a new tweet arrives, it retrieves the top 3 most similar historical threads to ground the LLM on *how Spotify handles this specific issue*.

**LLM stack:** Agent, auto-labeler, and judge all use **DeepSeek** (`deepseek-chat`) via an OpenAI-compatible API.

**Structured Output:**
The LLM is strictly constrained to output JSON, ensuring downstream systems can programmatically parse the intent, escalation boolean, and draft reply.

**Escalation Logic:**
The system is explicitly prompted to prioritize recall over precision for escalations. Missing a hacking report is a critical failure; over-escalating a minor complaint is an acceptable inefficiency.

---

## 3. Evaluation Design

Evaluating generative systems requires rigorous, automated, and human-aligned metrics.

**Golden Evaluation Set:**
*   **Size:** 200 examples.
*   **Sampling:** We extracted all "first-contact" user tweets. To ensure topical diversity and avoid evaluating on 200 identical "app crashed" tweets, we vectorized the tweets using TF-IDF and clustered them into 15 distinct topics using KMeans. We then sampled proportionally from these clusters.
*   **Labeling:** We used DeepSeek (`deepseek-chat`) for initial auto-labeling (Intent, Escalation), with structure suitable for human review.

**Baselines:**
To prove the value of the RAG architecture, we evaluate against two baselines:
1.  **Trivial Baseline:** A hardcoded agent that always predicts "General_Inquiry", always escalates, and provides no draft reply. This establishes the absolute floor.
2.  **Zero-Shot Agent:** An LLM with the persona prompt but *no* retrieved historical context.

**LLM-as-Judge Rubric:**
We evaluate draft replies with DeepSeek (`deepseek-chat`) as judge, scoring on a 1-5 scale:
*   **Empathy & Tone:** Brand alignment.
*   **Helpfulness:** Actionability and specific troubleshooting.
*   **Safety:** Avoidance of hallucinations or false promises.

**Judge Calibration:**
Manual human grades on **30** RAG replies that also had LLM-judge scores (`outputs/human_agreement.json`):

| Dimension | Cohen's κ | Pearson r | MAE | Exact match |
| :--- | ---: | ---: | ---: | ---: |
| Empathy | 0.27 | 0.70 | 0.37 | 63% |
| Helpfulness | 0.48 | 0.81 | 0.33 | 67% |
| Safety | 0.46 | 0.84 | 0.10 | 90% |
| Overall (rounded) | 0.27 | 0.68 | 0.37 | 63% |

Agreement is moderate: the judge is usable as a ranking signal but tends to score slightly higher/more uniformly than the human reviewer (especially empathy). Interactive re-grading: `python scripts/run_human_eval.py`.

---

## 4. Results

**Run config:** 200 golden examples · DeepSeek `deepseek-chat` · local TF-IDF RAG · LLM judge every 5th generative reply (**n=40**).  
Artifacts: `outputs/metrics_summary.json`, `outputs/comparison.md`.

### 4.1 Intent Classification

| System | Accuracy | F1 Macro | F1 Weighted |
| :--- | :--- | :--- | :--- |
| Trivial Baseline | 0.15 | 0.033 | 0.039 |
| Zero-Shot LLM | 0.89 | 0.889 | 0.890 |
| **RAG Agent** | **0.88** | **0.877** | **0.880** |

### 4.2 Escalation Decision

| System | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| Trivial Baseline | 0.09 | 1.000 | 0.165 |
| Zero-Shot LLM | 0.205 | 1.000 | 0.340 |
| **RAG Agent** | **0.842** | **0.889** | **0.865** |

### 4.3 Reply Quality (LLM Judge, n=40)

| System | Empathy (/5) | Helpfulness (/5) | Safety (/5) | Overall (/5) |
| :--- | :--- | :--- | :--- | :--- |
| Zero-Shot LLM | 4.30 | 3.90 | 4.60 | 4.27 |
| **RAG Agent** | **4.80** | **4.33** | **4.95** | **4.69** |

### 4.4 Human–Judge Alignment

Manual review of 30 judged RAG examples (`outputs/human_agreement.json`):

| Dimension | Cohen's κ | Pearson r | MAE |
| :--- | ---: | ---: | ---: |
| Empathy | 0.27 | 0.70 | 0.37 |
| Helpfulness | 0.48 | 0.81 | 0.33 |
| Safety | 0.46 | 0.84 | 0.10 |
| Overall | 0.27 | 0.68 | 0.37 |

---

## 5. Failure Analysis (Observed from RAG results)

From this run, RAG’s main win is **escalation precision** (0.84 vs 0.21 zero-shot) with still-high recall (0.89). Intent F1 is slightly *below* zero-shot — retrieval does not automatically improve triage when both systems share the same LLM labeler family.

### Real failure cases (from `outputs/results_RAG_Agent.json`)

1. **Hallucinated “undo thumbs-down”**
   - **Tweet:** “I accidentally thumbs-downed one of my favourite songs in a station, how can I undo it?”
   - **Truth:** `Feature_Request` · **Pred:** `Playback_Issue`
   - **Draft:** Told the user to tap thumbs-down again to undo.
   - **Human reply:** Feature isn’t possible; pointed to a feedback vote link.
   - **Why it failed:** Parametric “helpful” advice overrode the real product limitation; retrieval didn’t surface a “not currently possible” precedent strongly enough.

2. **Wrong diagnosis on Samsung TV**
   - **Tweet:** “not working in samsung tv. Any known issues. Every other app is working perfect.”
   - **Truth:** `Content_Availability` · **Pred:** `Playback_Issue`
   - **Draft:** Generic reinstall / troubleshooting steps.
   - **Human reply:** Spotify was removing the Samsung TV app (policy/content platform change).
   - **Why it failed:** Intent lookalike (playback) + retrieval of generic device-fix threads instead of deprecation notices.

3. **Missed escalation on extreme frustration**
   - **Tweet:** “and with no customer service line, you guys suck!”
   - **Truth escalate:** `True` · **Pred escalate:** `False`
   - **Draft:** Soft apology + ask for DM/email.
   - **Why it failed:** Insult-only / channel-complaint language didn’t trip security/billing keywords; recall-biased policy still missed pure rage-without-issue cases.

### Other recurring modes

1. **Over-Escalation on Mild Frustration:**
    *   *Description:* Casual profanity or payment wording triggers escalation even when golden labels say no.
    *   *Hypothesis:* Prompt rules and billing keywords are coarse for multilingual / slangy tweets.
2. **Intent Confusion (Playback vs. App Crash):**
    *   *Description:* Freeze-while-playing and similar cases flip between `Playback_Issue` and `App_Bug_Crash`.
    *   *Hypothesis:* Taxonomy granularity is too fine for overlapping symptoms.
3. **Tone Mismatch on Serious Issues:**
    *   *Description:* Casual “Hey there!” tone on high-stress account/library loss.
    *   *Hypothesis:* Brand-persona instructions override situational severity.

---

## 6. "What is Misleading About My Headline Number?"

This section outlines the systemic biases and limitations in this evaluation:

*   **Intent F1 looks “solved” (~0.89):** Zero-shot and RAG are labeled by the same LLM family as the golden auto-labeler, so high F1 partly measures self-consistency, not independent human triage quality. Also, RAG did *not* improve intent over zero-shot in this run.
*   **Judge Preference for Verbosity:** The LLM-as-judge inherently prefers longer, highly structured, and polite responses. However, real `SpotifyCares` agents are often terse and highly efficient ("DM us your email address"). The judge might penalize realistic, concise answers in favor of verbose AI-speak.
*   **Single-Turn Myopia:** We only evaluate the first response. A draft reply might score 5/5 for helpfulness but completely derail the conversation in turn two. We miss the most valuable signal: does the interaction actually resolve the user's issue?
*   **Survivorship Bias in the Dataset:** The Kaggle dataset only contains conversations that *happened*. It does not include users who were too frustrated to tweet, or whose tweets were ignored.
*   **The RAG Corpus IS the Training Set:** Because we retrieve from the same distribution we sample our eval set from, retrieval is highly likely to find in-distribution examples. This makes the evaluation optimistic compared to deploying the agent into a live stream of novel, unseen issues.

---

## 7. What I'd Do With One More Week

*   **Fine-Tune a Lightweight Model:** Distill successful DeepSeek RAG traces into a smaller OSS model (e.g. Llama 3 8B) for latency/cost.
*   **Multi-Turn Thread Handling:** Implement conversation state tracking to evaluate full resolution trajectories rather than single-turn triage.
*   **Adversarial Evaluation Set:** Inject deliberately confusing queries, prompt injections, and out-of-domain edge cases into the golden set to test robustness.
*   **A/B Test Escalation Thresholds:** Implement a tunable confidence threshold for the escalation decision and plot a Precision-Recall curve to find the optimal operating point.
*   **Dense Retriever Option:** Swap TF-IDF for local sentence-transformers embeddings if recall@3 plateaus on paraphrases.

---

## 8. Decision Log

*   **Decision:** Chose SpotifyCares over AppleSupport. — **Why:** Apple's responses are often a generic "DM us your device details." Spotify has more diverse troubleshooting logic and a distinct tone, making it a better test of LLM capabilities.
*   **Decision:** RAG architecture over Fine-Tuning. — **Why:** Support policies change. Updating a JSON corpus is trivial; re-training a model is expensive and slow.
*   **Decision:** TF-IDF + KMeans clustering for eval sampling. — **Why:** Random sampling yields 80% "app crashed" tweets. Clustering ensures diversity without dense-embedding cost on the full pool.
*   **Decision:** Local TF-IDF + numpy cosine over FAISS / cloud embeddings. — **Why:** Corpus cap (~5,000) is small; avoids embedding API dependency so the project runs with a DeepSeek-only key.
*   **Decision:** DeepSeek (`deepseek-chat`) for agent, labeling, and judge. — **Why:** OpenAI-compatible, cost-effective; same family for generate+judge is a known bias we disclose in human κ.
*   **Decision:** Enforced JSON structured output. — **Why:** Crucial for automated metrics. Parsing regex from free-text LLM outputs is brittle.
*   **Decision:** 200 samples for the Golden Set. — **Why:** Balances statistical significance with the practical reality of manual review time.
*   **Decision:** Evaluated first-contact pairs only. — **Why:** Avoids the compounding errors of managing thread state, isolating the evaluation to triage and initial response generation.
*   **Decision:** Temperature 0 for LLM calls. — **Why:** Prioritized reproducibility over creative diversity for scientific evaluation.
*   **Decision:** Separate judge prompt/role (+ human κ check). — **Why:** Reduces (does not eliminate) self-preference; we report moderate agreement honestly.
*   **Decision:** Escalation rules optimized for Recall. — **Why:** In real-world support, missing a critical crisis (hack, legal) is catastrophically worse than having a human review a false positive.
*   **Decision:** Capped RAG corpus at 5,000. — **Why:** Enough density for a POC without indexing the entire brand history.
*   **Decision:** Auto-labeling with human review structure. — **Why:** Faster than labeling 200 from scratch; we treat intent F1 as partly self-consistency.
*   **Decision:** Single orchestration script (`run_pipeline.py`). — **Why:** Maximizes reproducibility. Reviewers can run the entire project end-to-end with one command.
*   **Decision:** Included "Misleading Numbers" section. — **Why:** Intellectual honesty is the strongest signal of a mature engineering mindset. Every metric has flaws; acknowledging them builds trust.
