import pandas as pd
import jiwer
from pathlib import Path

# Load your merged CSV
input_csv = "/Users/vkodali/Downloads/whisper_edge_transcripts_with_originals.csv"
df = pd.read_csv(input_csv)

# Define normalization
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

alignment_visualizations = []

# Loop through each row
for idx, row in df.iterrows():
    ref = transformation(str(row["original_transcript"]))
    hyp = transformation(str(row["edge_transcript"]))

    # process_words expects a list of sentences
    out = jiwer.process_words([ref], [hyp])

    # Get alignment visualization as a string
    alignment_str = jiwer.visualize_alignment(out)

    alignment_visualizations.append(alignment_str)

# Add alignment as a new column
df["alignment_visualization"] = alignment_visualizations

# Save updated CSV
output_path = Path.cwd() / "data" / "whisper_edge_transcripts_with_alignment.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

downloads_path = Path.home() / "Downloads" / "whisper_edge_transcripts_with_alignment.csv"
df.to_csv(downloads_path, index=False)

print("Saved CSV in Downloads folder:", downloads_path)
print("Saved CSV with alignment visualizations to:", output_path)
