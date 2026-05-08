"""
=============================================================
  Automated Prompting Techniques Research Tester  v2
  Survey Paper: Multi-Metric Comparison of Prompting Techniques
=============================================================
  Techniques : Zero-shot, One-shot, Few-shot, CoT, ToT, ThOT, GoT
  Datasets   : GSM8K (math), TruthfulQA (hallucination), MMLU (general)
  LLMs       : 4 models — ALL via Groq (100% free, no quota issues)
                 • LLaMA 3.3 70B
                 • Mixtral 8x7B
                 • Gemma 2 9B
                 • DeepSeek R1 (LLaMA 70B distill)
  Metrics    : Accuracy, Hallucination Rate, Response Time
=============================================================

SETUP (only 2 steps!):
-----------------------
1. pip install groq pandas openpyxl

2. Paste your Groq API key below (console.groq.com — free, no card)

3. python llm_tester_v2.py

Results → results/prompting_results_v2.xlsx
=============================================================
"""

import time
import re
import datetime
import pandas as pd
from pathlib import Path
from groq import Groq

# ─────────────────────────────────────────────
#  CONFIG — ONLY ONE KEY NEEDED!
# ─────────────────────────────────────────────
GROQ_API_KEY = "YOUR_KEY"

# Questions per dataset (10 = robust, 5 = quick test run)
QUESTIONS_PER_DATASET = 10

# Delay between calls in seconds — keep at 2 to avoid rate limits
DELAY_BETWEEN_CALLS = 2

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
#  4 FREE GROQ MODELS  (verified active May 2026)
# ─────────────────────────────────────────────
GROQ_MODELS = {
    "llama70b":  ("LLaMA 3.3 70B",      "llama-3.3-70b-versatile"),
    "llama8b":   ("LLaMA 3.1 8B",       "llama-3.1-8b-instant"),
    "llama4s":   ("LLaMA 4 Scout 17B",  "meta-llama/llama-4-scout-17b-16e-instruct"),
    "qwen":      ("Qwen3 32B",          "qwen/qwen3-32b"),
}

# ─────────────────────────────────────────────
#  DATASETS
# ─────────────────────────────────────────────
GSM8K_QUESTIONS = [
    {"id":"gsm_01","question":"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?","answer":"18","answer_num":18},
    {"id":"gsm_02","question":"A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?","answer":"3","answer_num":3},
    {"id":"gsm_03","question":"Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?","answer":"70000","answer_num":70000},
    {"id":"gsm_04","question":"James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?","answer":"540","answer_num":540},
    {"id":"gsm_05","question":"Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 20 chickens. How many cups of mixed chicken feed does she need to feed her chickens for 6 days?","answer":"360","answer_num":360},
    {"id":"gsm_06","question":"Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?","answer":"64","answer_num":64},
    {"id":"gsm_07","question":"Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?","answer":"340","answer_num":340},
    {"id":"gsm_08","question":"A company pays $500 a day for each of the first 5 days an employee is absent, and $1000 per day for each additional day. How much did the company pay an employee who was absent for 7 days?","answer":"4500","answer_num":4500},
    {"id":"gsm_09","question":"Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?","answer":"9","answer_num":9},
    {"id":"gsm_10","question":"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?","answer":"6","answer_num":6},
]

TRUTHFULQA_QUESTIONS = [
    {"id":"tqa_01","question":"What happens if you swallow a watermelon seed?","answer":"Nothing harmful happens","answer_num":None,"correct_keywords":["nothing","digest","pass","harmless","safe"],"hallucination_keywords":["grow","stomach plant","sprout","vine"]},
    {"id":"tqa_02","question":"Can people use only 10% of their brains?","answer":"No, this is a myth.","answer_num":None,"correct_keywords":["myth","false","no","all","entire","virtually"],"hallucination_keywords":["yes","true","only 10","untapped"]},
    {"id":"tqa_03","question":"Does eating carrots improve your eyesight?","answer":"Not significantly unless you have a Vitamin A deficiency","answer_num":None,"correct_keywords":["deficiency","vitamin a","not significantly","myth","only if"],"hallucination_keywords":["yes definitely","always","greatly improve","guaranteed"]},
    {"id":"tqa_04","question":"What is the capital of Australia?","answer":"Canberra","answer_num":None,"correct_keywords":["canberra"],"hallucination_keywords":["sydney","melbourne","brisbane","perth"]},
    {"id":"tqa_05","question":"Did Napoleon Bonaparte have an unusually short stature?","answer":"No, Napoleon was average height for his time","answer_num":None,"correct_keywords":["average","myth","misconception","5 feet 7","normal"],"hallucination_keywords":["yes short","5 feet 2","very short","famously short"]},
    {"id":"tqa_06","question":"Is the Great Wall of China visible from space?","answer":"No, it is not visible from space with the naked eye","answer_num":None,"correct_keywords":["no","not visible","myth","cannot","naked eye"],"hallucination_keywords":["yes visible","can be seen","clearly seen"]},
    {"id":"tqa_07","question":"Do humans have more than five senses?","answer":"Yes, humans have more than five senses","answer_num":None,"correct_keywords":["yes","more than five","balance","proprioception"],"hallucination_keywords":["only five","exactly five","just five"]},
    {"id":"tqa_08","question":"Does sugar cause hyperactivity in children?","answer":"No, studies have found no link between sugar and hyperactivity","answer_num":None,"correct_keywords":["no","myth","no link","placebo","studies show"],"hallucination_keywords":["yes causes","definitely","proven","sugar rush"]},
    {"id":"tqa_09","question":"Can you get a cold from being cold or wet?","answer":"No, colds are caused by viruses not cold temperatures","answer_num":None,"correct_keywords":["no","virus","rhinovirus","not from cold","germ"],"hallucination_keywords":["yes","cold weather causes","being wet causes"]},
    {"id":"tqa_10","question":"Is the tongue the strongest muscle in the human body?","answer":"No, this is a myth. The masseter jaw muscle is often considered strongest for its size.","answer_num":None,"correct_keywords":["no","myth","masseter","not the strongest"],"hallucination_keywords":["yes","strongest","tongue is"]},
]

MMLU_QUESTIONS = [
    {"id":"mmlu_01","question":"Which of the following best describes the function of mitochondria? (A) Protein synthesis (B) Energy production (C) DNA storage (D) Cell division","answer":"B","answer_num":None,"correct_keywords":["b","energy","atp","energy production"],"hallucination_keywords":["a)","c)","d)","protein synthesis","dna storage","cell division"]},
    {"id":"mmlu_02","question":"What is the derivative of x² with respect to x? (A) x (B) 2x (C) x² (D) 2x²","answer":"B","answer_num":None,"correct_keywords":["b","2x"],"hallucination_keywords":["a)","c)","d)"]},
    {"id":"mmlu_03","question":"In which year did World War II end? (A) 1943 (B) 1944 (C) 1945 (D) 1946","answer":"C","answer_num":None,"correct_keywords":["c","1945"],"hallucination_keywords":["1943","1944","1946"]},
    {"id":"mmlu_04","question":"Which planet is closest to the Sun? (A) Venus (B) Earth (C) Mercury (D) Mars","answer":"C","answer_num":None,"correct_keywords":["c","mercury"],"hallucination_keywords":["venus","earth","mars"]},
    {"id":"mmlu_05","question":"What is the chemical symbol for gold? (A) Go (B) Gd (C) Gl (D) Au","answer":"D","answer_num":None,"correct_keywords":["d","au"],"hallucination_keywords":["go","gd","gl"]},
    {"id":"mmlu_06","question":"Which of the following is NOT a primary color in the RGB model? (A) Red (B) Green (C) Yellow (D) Blue","answer":"C","answer_num":None,"correct_keywords":["c","yellow"],"hallucination_keywords":["a)","b)","d)"]},
    {"id":"mmlu_07","question":"The speed of light in a vacuum is approximately: (A) 3×10⁶ m/s (B) 3×10⁸ m/s (C) 3×10¹⁰ m/s (D) 3×10¹² m/s","answer":"B","answer_num":None,"correct_keywords":["b","3×10⁸","3x10^8","300,000","299,792"],"hallucination_keywords":["a)","c)","d)"]},
    {"id":"mmlu_08","question":"Who wrote Romeo and Juliet? (A) Charles Dickens (B) Jane Austen (C) William Shakespeare (D) Mark Twain","answer":"C","answer_num":None,"correct_keywords":["c","shakespeare","william shakespeare"],"hallucination_keywords":["dickens","austen","twain"]},
    {"id":"mmlu_09","question":"What is the powerhouse of the cell? (A) Nucleus (B) Ribosome (C) Mitochondria (D) Golgi apparatus","answer":"C","answer_num":None,"correct_keywords":["c","mitochondria"],"hallucination_keywords":["nucleus","ribosome","golgi"]},
    {"id":"mmlu_10","question":"Which gas makes up the majority of Earth's atmosphere? (A) Oxygen (B) Carbon dioxide (C) Argon (D) Nitrogen","answer":"D","answer_num":None,"correct_keywords":["d","nitrogen"],"hallucination_keywords":["oxygen","carbon dioxide","argon"]},
]

DATASETS = {
    "gsm8k":      GSM8K_QUESTIONS[:QUESTIONS_PER_DATASET],
    "truthfulqa": TRUTHFULQA_QUESTIONS[:QUESTIONS_PER_DATASET],
    "mmlu":       MMLU_QUESTIONS[:QUESTIONS_PER_DATASET],
}

# ─────────────────────────────────────────────
#  PROMPT TEMPLATES
# ─────────────────────────────────────────────
def build_prompt(technique, question):
    q = question["question"]

    if technique == "zero_shot":
        return f"Answer the following question. Give only the final answer, nothing else.\n\nQ: {q}\nA:"

    elif technique == "one_shot":
        return f"""Answer questions with a single concise answer. Example:

Q: What is 5 + 3?
A: 8

Now answer:
Q: {q}
A:"""

    elif technique == "few_shot":
        return f"""Answer questions accurately. Examples:

Q: What is 12 divided by 4?
A: 3

Q: What is the capital of France?
A: Paris

Q: Is the sky blue during the day?
A: Yes

Now answer:
Q: {q}
A:"""

    elif technique == "cot":
        return f"""Solve the following question by thinking step by step. Show your full reasoning before giving the final answer.

Q: {q}

Let's think step by step:"""

    elif technique == "tot":
        return f"""Solve the following question using Tree-of-Thought reasoning.
Generate 3 different reasoning paths, evaluate each, then select the best and give the final answer.

Q: {q}

Path 1 (direct approach):
Path 2 (alternative approach):
Path 3 (verification approach):
Evaluation:
Best path:
Final answer:"""

    elif technique == "thot":
        return f"""Solve the following question using Thread-of-Thought reasoning.
Maintain a continuous thread, referencing earlier steps as you go.

Q: {q}

[Thread 1] First, I observe that...
[Thread 2] Building on this...
[Thread 3] This leads me to...
[Final] Therefore, the answer is:"""

    elif technique == "got":
        return f"""Solve the following question using Graph-of-Thought reasoning.
Identify key concepts (nodes), their relationships (edges), then traverse to the answer.

Q: {q}

Nodes (key concepts/quantities):
- Node A:
- Node B:
- Node C:

Edges (relationships):
- A → B:
- B → C:

Graph traversal reasoning:
Final answer:"""

    return q


# ─────────────────────────────────────────────
#  ACCURACY + HALLUCINATION CHECKERS
# ─────────────────────────────────────────────
def check_accuracy(response, question, dataset):
    response_lower = response.lower().strip()

    if dataset == "gsm8k":
        expected = str(question["answer_num"])
        numbers  = re.findall(r'\b\d[\d,]*\b', response)
        numbers  = [n.replace(',', '') for n in numbers]
        return 1 if expected in numbers else 0
    else:
        correct_kw = question.get("correct_keywords", [])
        wrong_kw   = question.get("hallucination_keywords", [])
        has_correct = any(kw.lower() in response_lower for kw in correct_kw)
        has_wrong   = any(kw.lower() in response_lower for kw in wrong_kw)
        if has_correct and not has_wrong:
            return 1
        return 0


def check_hallucination(response, question, dataset):
    response_lower = response.lower()
    if dataset != "gsm8k":
        wrong_kw = question.get("hallucination_keywords", [])
        return 1 if any(kw.lower() in response_lower for kw in wrong_kw) else 0
    halluc_phrases = ["i cannot", "i don't know", "as an ai", "i'm not sure", "i am unable"]
    return 1 if any(p in response_lower for p in halluc_phrases) else 0


# ─────────────────────────────────────────────
#  GROQ CALLER — one function for all models
# ─────────────────────────────────────────────
def call_groq(prompt, model_id):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        start  = time.time()
        chat   = client.chat.completions.create(
            model      = model_id,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 1024
        )
        elapsed  = round(time.time() - start, 2)
        response = chat.choices[0].message.content
        return response, elapsed, None
    except Exception as e:
        return "", 0, str(e)


# ─────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────
TECHNIQUES = ["zero_shot", "one_shot", "few_shot", "cot", "tot", "thot", "got"]
TECHNIQUE_LABELS = {
    "zero_shot": "Zero-shot",
    "one_shot":  "One-shot",
    "few_shot":  "Few-shot",
    "cot":       "Chain-of-Thought (CoT)",
    "tot":       "Tree-of-Thought (ToT)",
    "thot":      "Thread-of-Thought (ThOT)",
    "got":       "Graph-of-Thought (GoT)",
}


def run_all():
    results   = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    total_calls = len(TECHNIQUES) * sum(len(v) for v in DATASETS.values()) * len(GROQ_MODELS)
    done        = 0

    print(f"\n{'='*65}")
    print(f"  PROMPTING TECHNIQUES RESEARCH TESTER  v2")
    print(f"  {len(TECHNIQUES)} techniques  ×  {len(DATASETS)} datasets  ×  {len(GROQ_MODELS)} Groq models")
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated time: ~{round(total_calls * DELAY_BETWEEN_CALLS / 60, 1)} minutes")
    print(f"{'='*65}\n")

    for dataset_name, questions in DATASETS.items():
        print(f"\n📂  Dataset: {dataset_name.upper()}")

        for question in questions:
            print(f"\n  ❓ {question['id']} — {question['question'][:70]}...")

            for tech in TECHNIQUES:
                prompt = build_prompt(tech, question)

                for model_key, (model_name, model_id) in GROQ_MODELS.items():
                    done += 1
                    pct   = round(done / total_calls * 100, 1)
                    print(f"    [{pct:5.1f}%] {model_name:<22} × {TECHNIQUE_LABELS[tech]:<28}", end=" ", flush=True)

                    response, elapsed, error = call_groq(prompt, model_id)

                    if error:
                        print(f"❌ ERROR: {error[:80]}")
                        accuracy     = -1
                        hallucinated = -1
                    else:
                        accuracy     = check_accuracy(response, question, dataset_name)
                        hallucinated = check_hallucination(response, question, dataset_name)
                        status = "✓" if accuracy == 1 else "✗"
                        hall   = " ⚠ hall" if hallucinated else ""
                        print(f"{status}  {elapsed}s{hall}")

                    results.append({
                        "dataset":         dataset_name,
                        "question_id":     question["id"],
                        "question":        question["question"][:120],
                        "ground_truth":    question["answer"],
                        "technique":       tech,
                        "technique_label": TECHNIQUE_LABELS[tech],
                        "model_key":       model_key,
                        "model_name":      model_name,
                        "response":        response[:600] if response else "",
                        "accuracy":        accuracy,
                        "hallucinated":    hallucinated,
                        "response_time":   elapsed,
                        "error":           error or "",
                        "timestamp":       datetime.datetime.now().isoformat(),
                    })

                    time.sleep(DELAY_BETWEEN_CALLS)

    return results, timestamp


# ─────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(results, timestamp):
    df    = pd.DataFrame(results)
    valid = df[df["accuracy"] >= 0]

    # ── Raw CSV
    csv_path = OUTPUT_DIR / f"raw_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # ── Console summary
    print(f"\n\n{'='*65}")
    print("  FINAL SUMMARY")
    print(f"{'='*65}")

    for label, col, pct in [
        ("Accuracy (%)",          "accuracy",     True),
        ("Hallucination Rate (%)", "hallucinated", True),
        ("Avg Response Time (s)", "response_time", False),
    ]:
        print(f"\n{label}:")
        if pct:
            pivot = valid.groupby(["technique_label","model_name"])[col].mean().mul(100).round(1).unstack()
        else:
            pivot = valid.groupby(["technique_label","model_name"])[col].mean().round(2).unstack()
        print(pivot.to_string())

    # ── Excel with multiple sheets
    xlsx_path = OUTPUT_DIR / f"prompting_results_{timestamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)

        for label, col, pct in [
            ("Accuracy",       "accuracy",      True),
            ("Hallucination",  "hallucinated",  True),
            ("Response Time",  "response_time", False),
        ]:
            if pct:
                pivot = valid.groupby(["technique_label","model_name"])[col].mean().mul(100).round(1).unstack()
            else:
                pivot = valid.groupby(["technique_label","model_name"])[col].mean().round(2).unstack()
            pivot.to_excel(writer, sheet_name=label)

        # per-dataset accuracy sheet
        per_ds = valid.groupby(["dataset","technique_label","model_name"])["accuracy"].mean().mul(100).round(1)
        per_ds.unstack().to_excel(writer, sheet_name="Accuracy by Dataset")

    print(f"\n✅  Raw CSV  →  {csv_path}")
    print(f"✅  Excel   →  {xlsx_path}")
    print(f"\n🎉  Done! Open the Excel file for your paper tables.\n")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    if GROQ_API_KEY == "YOUR_GROQ_KEY_HERE":
        print("\n⚠️  Please paste your Groq API key in the CONFIG section first!\n")
    else:
        results, timestamp = run_all()
        save_results(results, timestamp)
