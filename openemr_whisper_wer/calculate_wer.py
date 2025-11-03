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

# Aggregate WER
total_ref = " ".join(refs)
total_hyp = " ".join(hyps)

overall_wer = jiwer.wer(total_ref, total_hyp)

print("Overall WER (aggregate) for Xenova/whisper-tiny.en:", overall_wer)

"------------------------------------------------------------------------------------"

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

# Aggregate WER
total_ref = " ".join(refs)
total_hyp = " ".join(hyps)

overall_wer = jiwer.wer(total_ref, total_hyp)

print("Overall WER (aggregate) for Xenova/whisper-base.en:", overall_wer)

"------------------------------------------------------------------------------------"

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

# Aggregate WER
total_ref = " ".join(refs)
total_hyp = " ".join(hyps)

overall_wer = jiwer.wer(total_ref, total_hyp)

print("Overall WER (aggregate) for Xenova/whisper-edge.en:", overall_wer)