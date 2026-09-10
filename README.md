# SpotifyCares AI Support Agent

End-to-end customer support AI agent for **SpotifyCares** on the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. Classifies intent, decides escalation, drafts brand-aligned replies, and evaluates with baselines + LLM-as-judge.

## Quick Start

```bash
pip install -r requirements.txt

# Create .env (DeepSeek OpenAI-compatible API)
# OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://api.deepseek.com
# OPENAI_MODEL=deepseek-chat
# JUDGE_MODEL=deepseek-chat

python scripts/download_data.py
python run_pipeline.py --step eval --no-judge   # reuse cached golden/RAG (~few min)
# Or full rebuild:
python run_pipeline.py --step all               # data + golden + RAG + eval + judge
```

Interactive demo: `python scripts/demo.py`  
Human–judge agreement: `python scripts/run_human_eval.py`

## Architecture

```text
User Tweet
    │
    ▼
[ Intent + Escalation ] ─── 8-way taxonomy, recall-biased escalation
    │
    ▼
[ RAG Retriever ] ─────── top-3 similar historical SpotifyCares threads (local TF-IDF)
    │
    ▼
[ LLM Reply Generator ] ─ DeepSeek chat (JSON structured output)
    │
    ▼
[ Structured Output ] ─── intent, should_escalate, reason, draft_reply
```

## Problem Framing

- **Target Brand**: SpotifyCares
- **Why Spotify**: Diverse intents (bugs, billing, content), distinct brand voice, manageable volume.
- **What "Good" Means**: Accurate triage, safe escalation (recall > precision), replies that match SpotifyCares tone.
- **Non-goals**: Multi-turn dialogue, live Twitter API, fine-tuning.

## Intent Taxonomy

1. Account_Access  
2. Billing_Payment  
3. Playback_Issue  
4. App_Bug_Crash  
5. Content_Availability  
6. Subscription_Plan  
7. Feature_Request  
8. General_Inquiry  

## Evaluation Framework

- **Golden set**: 200 examples — TF-IDF + KMeans diversity sampling, LLM auto-labels. See [`eval/GOLDEN_SET.md`](eval/GOLDEN_SET.md).
- **Systems**: Trivial baseline (always escalate) · Zero-shot LLM · RAG agent.
- **Metrics**: Intent F1 (macro/weighted), Escalation P/R/F1, LLM-as-judge (empathy / helpfulness / safety).
- **Judge alignment**: Manual grades on 30 RAG replies → overall κ ≈ **0.27**, helpfulness κ ≈ **0.48**, safety κ ≈ **0.46** (`outputs/human_agreement.json`).

## Results

Eval on **200** golden examples. LLM-as-judge sampled every 5th reply (**n=40** per generative system). Models: DeepSeek `deepseek-chat`; retrieval: local TF-IDF.

| System        | Intent F1 (M) | Esc. Precision | Esc. Recall | Esc. F1 | Judge Overall |
|---------------|---------------|----------------|-------------|---------|---------------|
| Trivial       | 0.033         | 0.09           | 1.00        | 0.17    | —             |
| Zero-Shot     | 0.889         | 0.21           | 1.00        | 0.34    | 4.27 / 5      |
| RAG Agent     | 0.877         | 0.84           | 0.89        | 0.86    | 4.69 / 5      |

Headline takeaway: RAG does **not** win on intent F1 vs zero-shot (near-tied), but clearly wins on **escalation precision** and **reply quality** under the judge.

Full tables: [`REPORT.md`](REPORT.md), [`outputs/comparison.md`](outputs/comparison.md), [`outputs/metrics_summary.json`](outputs/metrics_summary.json).

## Failure Analysis

### Real cases from the RAG eval run

1. **Hallucinated undo for thumbs-down** — User asked how to undo a thumbs-down; agent said “tap again,” but the human reply was “not currently possible” + vote link. Also mis-labeled as `Playback_Issue` instead of `Feature_Request`.
2. **Samsung TV treated as a playback bug** — Agent suggested reinstall steps; human replied that Spotify was removing the Samsung TV app (`Content_Availability` / platform policy).
3. **Missed escalation on pure rage** — “no customer service line, you guys suck!” labeled escalate=`True`, model answered without escalating.

### Other modes
Sarcasm/tone mismatch, multi-intent single-label collisions, and over-escalation on vague/multilingual billing wording.

## What is Misleading About My Headline Number?

- **Survivorship bias**: only threads where the brand replied publicly.  
- **Auto-labels as ground truth**: agent and labeler can share LLM biases → inflated agreement.  
- **LLM-as-judge verbosity bias**: prefers longer replies than real Twitter agents.  
- **Single-turn eval**: ignores multi-turn resolution.  
- **In-distribution RAG**: corpus and eval drawn from the same historical pool → optimistic retrieval.

## What I'd Do With One More Week

- Fine-tune a small OSS model on successful RAG traces.  
- Multi-turn state.  
- Adversarial / slang-heavy golden examples.  
- Escalation PR curves.  
- Intent-aware retrieval.

## Decision Log

- SpotifyCares over AppleSupport — more diverse intents + tone.  
- RAG over fine-tuning — policies change; corpus updates are cheap.  
- **Local TF-IDF retrieval** (not cloud embeddings) — runs with DeepSeek-only API keys; corpus ~5k is small enough.  
- TF-IDF clustering for golden sampling — diversity without embedding cost.  
- DeepSeek for agent + judge — cost-effective OpenAI-compatible API.  
- JSON structured output — reliable metrics parsing.  
- 200 golden examples — annotation cost vs significance.  
- First-contact pairs only — avoids multi-turn state.  
- Temperature 0 — reproducibility.  
- Escalation recall > precision — missed crises cost more than over-escalate.  
- Separate judge prompt/model role — reduce self-preference bias.

## Project Structure

```text
sharp-heisenberg/
├── README.md
├── REPORT.md
├── requirements.txt
├── .env                    # API key (gitignored)
├── run_pipeline.py
├── data/                   # twcs.csv + RAG artifacts (gitignored large files)
├── eval/
│   ├── golden_eval.json    # 200-example golden set
│   └── GOLDEN_SET.md
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── data_loader.py
│   ├── embeddings.py       # local TF-IDF
│   ├── evaluator.py
│   ├── eval_runner.py
│   ├── golden_set_builder.py
│   ├── llm_client.py       # DeepSeek / OpenAI-compatible client
│   └── rag.py
├── scripts/
│   ├── download_data.py
│   ├── demo.py
│   └── run_human_eval.py
└── outputs/                # metrics reports
```

## License

MIT License
