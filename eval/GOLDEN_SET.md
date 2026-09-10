# Golden Evaluation Set

**File:** `eval/golden_eval.json`  
**Size:** 200 examples (within the 150–250 brief)

## How it was compiled

1. Start from all SpotifyCares **first-contact** pairs in the Kaggle Twitter CS dataset (~28k threads: first user tweet + first brand reply).
2. Vectorize tweets with **TF-IDF**, cluster into **15** topics with **KMeans**, and sample **proportionally** across clusters so the set is not dominated by one issue type (e.g. app crashes).
3. **Auto-label** each example with DeepSeek (`deepseek-chat`) for intent (8-way taxonomy) and escalation (recall-biased rules).
4. Labels live under each example’s `auto_labels` field; metadata in the JSON records intent distribution and escalation rate.

## Known limitations

Auto-labels are LLM-assisted ground truth — they can share biases with the agent under test. Treat intent/escalation metrics as upper-bound estimates relative to this labeling scheme.
