import pandas as pd

# Paths to your CSVs
edge_csv = "/Users/vkodali/Downloads/whisper_edge_transcripts_ds.csv"
reference_csv = "/Users/vkodali/Downloads/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/overview-of-recordings.csv"

# Load both CSVs
df_edge = pd.read_csv(edge_csv)
df_ref = pd.read_csv(reference_csv)

# Normalize filenames in both DataFrames to ensure they match properly
df_edge["file_name"] = df_edge["file_name"].str.strip().str.lower()
df_ref["file_name"] = df_ref["file_name"].str.strip().str.lower()

# Merge on the common 'file_name' column
df_merged = pd.merge(df_edge, df_ref[["file_name", "phrase"]], on="file_name", how="left")

# Rename the 'phrase' column to 'original_transcript'
df_merged.rename(columns={"phrase": "original_transcript"}, inplace=True)

# Save the merged CSV to Downloads
output_csv = "/Users/vkodali/Downloads/whisper_edge_transcripts_with_originals.csv"
df_merged.to_csv(output_csv, index=False)

print("✅ Merged CSV saved to:", output_csv)
print("Columns in merged CSV:", df_merged.columns.tolist())
print(df_merged.head())

import pandas as pd
from pathlib import Path
import jiwer

# Load the merged CSV
input_csv = "/Users/vkodali/Downloads/whisper_edge_transcripts_with_originals.csv"
df = pd.read_csv(input_csv)

# Define jiwer transformation pipeline
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# Normalize and compute WER for each row
refs = [transformation(str(r)) for r in df["original_transcript"]]
hyps = [transformation(str(h)) for h in df["edge_transcript"]]
wer_scores = [jiwer.wer(ref, hyp) for ref, hyp in zip(refs, hyps)]

# Add WER scores to dataframe
df["wer_edge"] = wer_scores

# Calculate average WER
average_wer = sum(wer_scores) / len(wer_scores)
print(f"\nAverage WER: {average_wer:.4f}")

# save updated CSV to ./data folder
output_path = Path.cwd() / "data" / "whisper_edge_transcripts_with_wer.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# Print summary
print("Saved CSV with WER scores to:", output_path)

# Save updated CSV to Downloads folder
downloads_path = Path.home() / "Downloads" / "whisper_edge_transcripts_with_wer.csv"
df.to_csv(downloads_path, index=False)
print("Saved CSV with WER scores to Downloads folder:", downloads_path)

print("Columns:", df.columns.tolist())
print(df[["file_name", "wer_edge"]].head(10))