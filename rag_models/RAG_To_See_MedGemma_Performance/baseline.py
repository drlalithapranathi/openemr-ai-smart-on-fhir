# ==============================
# PHASE 0: BASELINE EVALUATION
# ==============================
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Define input file paths
REFERENCE_PATH = os.path.join("data", "reference.txt")
REFERENCE_PATH_1 = os.path.join("data", "reference_1.txt")

# Load both text files
with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
    reference_text = f.read().strip()

with open(REFERENCE_PATH_1, "r", encoding="utf-8") as f:
    generated_text = f.read().strip()

# Compute BLEU score
bleu_score = sentence_bleu(
    [reference_text.split()],     # reference (list of tokens)
    generated_text.split()        # candidate (list of tokens)
)

# Compute ROUGE-L score
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = scorer.score(reference_text, generated_text)["rougeL"].fmeasure

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# Save evaluation results
RESULT_PATH = os.path.join("results", "baseline_result.txt")
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"BLEU Score: {bleu_score:.4f}\n")
    f.write(f"ROUGE-L Score: {rouge_l:.4f}\n")

print(f"Results saved to {RESULT_PATH}")
