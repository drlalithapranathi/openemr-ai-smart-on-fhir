import jiwer
import pandas as pd
from pathlib import Path

# Path to your CSV
data_dir = Path.cwd() / "data"
csv_path = data_dir / "conversation_dataset_ww_t.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Create a jiwer transformation
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# Normalize all references and hypotheses
refs = [transformation(r) for r in df["reference"]]
hyps = [transformation(h) for h in df["hypothesis"]]

# Store individual WERs in a list using a loop without zip
sentence_wers = []
for i in range(len(refs)):
    wer_score = jiwer.wer(refs[i], hyps[i])
    sentence_wers.append(wer_score)

# Calculate mean WER
average_wer = sum(sentence_wers) / len(sentence_wers)

print("Model name: Xenova/whisper-tiny.en")
print("Sentence-level WERs:", sentence_wers)
print("Average WER:", average_wer)

"---------------------------------------------------------------------------"
# Path to your CSV
data_dir = Path.cwd() / "data"
csv_path = data_dir / "conversation_dataset_ww_base.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Create a jiwer transformation
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# Normalize all references and hypotheses
refs = [transformation(r) for r in df["reference"]]
hyps = [transformation(h) for h in df["hypothesis"]]

# Store individual WERs in a list using a loop without zip
sentence_wers = []
for i in range(len(refs)):
    wer_score = jiwer.wer(refs[i], hyps[i])
    sentence_wers.append(wer_score)

# Calculate mean WER
average_wer = sum(sentence_wers) / len(sentence_wers)

print("Model name: Xenova/whisper-base.en")
print("Sentence-level WERs:", sentence_wers)
print("Average WER:", average_wer)
"---------------------------------------------------------------------------------------------------"
# Path to your CSV
data_dir = Path.cwd() / "data"
csv_path = data_dir / "conversation_dataset_w_edge.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Create a jiwer transformation
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# Normalize all references and hypotheses
refs = [transformation(r) for r in df["reference"]]
hyps = [transformation(h) for h in df["hypothesis"]]

# Store individual WERs
sentence_wers = []
for i in range(len(refs)):
    wer_score = jiwer.wer(refs[i], hyps[i])
    sentence_wers.append(wer_score)

# Calculate mean WER
average_wer = sum(sentence_wers) / len(sentence_wers)

print("Model name: Xenova/whisper-edge.en")
print("Sentence-level WERs:", sentence_wers)
print("Average WER:", average_wer)