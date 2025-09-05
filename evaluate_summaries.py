import json
import re
import time
from openai import OpenAI

# ========== CONFIG ==========
API_KEY = "input your key"
MODEL_NAME = "gpt-4-0125-preview"
ORIGINAL_FILE = "original_discharge_note.txt"
SUMMARY_FILE = "diverse_summaries_cleaned.json"
OUTPUT_FILE = "scored_summaries.json"
BEST_FILE = "best_summary.txt"

# ========== INITIALIZE ==========
client = OpenAI(api_key=API_KEY)

# ========== PROMPT TEMPLATE ==========
def build_prompt(original_text, summary_text):
    return f"""
Let's evaluate a medical summary on a Likert scale from 1 to 100, 1 being the worst and 100 being the best.

You will be given a hospital discharge note (original) and a generated summary. Evaluate the summary on four dimensions:

1. Clarity — Is it understandable and well-written?
2. Accuracy — Does it reflect the original facts correctly?
3. Coverage — Does it include all key medical content (diagnoses, meds, follow-ups)?
4. Completeness — Does the summary capture all major clinical events, diagnoses, treatments, and discharge instructions without omitting essential details?
5. Coherence — Is the summary logically structured and easy to follow without abrupt shifts or disjointed ideas?
6. Overall Quality — Is this a clinically usable, patient-appropriate summary?

Return your answer in the following format:

Clarity: [1–100]  
Accuracy: [1–100]  
Coverage: [1–100]  
Completeness: [1–100]
Coherence: [1–100]
Overall Quality: [1–100]  
Explanation: [brief rationale for each score]

Original Text:
{original_text}

Summary:
{summary_text}
"""

# ========== SCORE EXTRACTION ==========
def extract_scores(text):
    try:
        return {
            "clarity_score": int(re.search(r"Clarity:\s*(\d{1,3})", text).group(1)),
            "accuracy_score": int(re.search(r"Accuracy:\s*(\d{1,3})", text).group(1)),
            "coverage_score": int(re.search(r"Coverage:\s*(\d{1,3})", text).group(1)),
            "coherence_score": int(re.search(r"Coherence:\s*(\d{1,3})", text).group(1)),
            "completeness_score": int(re.search(r"Completeness:\s*(\d{1,3})", text).group(1)),
            "overall_quality_score": int(re.search(r"Overall Quality:\s*(\d{1,3})", text).group(1)),
        }
    except Exception as e:
        print("⚠️ Failed to extract one or more scores:", e)
        return {
            "clarity_score": None,
            "accuracy_score": None,
            "coverage_score": None,
            "coherence_score": None,
            "completeness_score": None,
            "overall_quality_score": None
        }

# ========== LOAD FILES ==========
with open(ORIGINAL_FILE, "r", encoding="utf-8") as f:
    original_text = f.read()

with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    summaries = json.load(f)

# ========== EVALUATE SUMMARIES ==========
for summary in summaries:
    print(f"🔍 Evaluating Summary {summary['summary_id']}...")

    prompt = build_prompt(original_text, summary["summary_text"])

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        reply = response.choices[0].message.content
        print(f"✅ Response received for Summary {summary['summary_id']}")
    except Exception as e:
        print(f"❌ API call failed for Summary {summary['summary_id']}: {e}")
        reply = ""

    scores = extract_scores(reply)
    print(f"Scores: {scores}")
    summary.update(scores)
    summary["gpt_evaluation_response"] = reply

    time.sleep(1.5)  # polite pause

# ========== SAVE SCORED OUTPUT ==========
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

print(f"\n🎯 Evaluation completed. Results saved to {OUTPUT_FILE}")

# ========== SELECT BEST SUMMARY ==========
def weighted_score(s):
    try:
        return (
            0.2 * s["clarity_score"] +
            0.2 * s["accuracy_score"] +
            0.2 * s["coverage_score"] +
            0.2 * s["coherence_score"] +
            0.1 * s["completeness_score"] +
            0.1 * s["overall_quality_score"]
        )
    except Exception as e:
        return -1

scored = [(s["summary_id"], weighted_score(s)) for s in summaries if weighted_score(s) >= 0]

if not scored:
    print("❌ No valid summaries with scores were found.")
else:
    best_summary_id, best_score = max(scored, key=lambda x: x[1])
    print(f"\n🏆 Best Summary: {best_summary_id} with weighted score {best_score:.2f}")

    best_summary = next((s for s in summaries if s["summary_id"] == best_summary_id), None)
    if best_summary:
        with open(BEST_FILE, "w", encoding="utf-8") as f:
            f.write(f"Summary {best_summary_id} (Score: {best_score:.2f})\n\n")
            f.write(best_summary["summary_text"])
        print(f"📄 Best summary saved to {BEST_FILE}")

