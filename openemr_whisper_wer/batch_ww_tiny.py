import csv

# Input and output file paths
input_file = "/Users/vkodali/IdeaProjects/word_error_rate_whisperweb/data/batch_transcribe_ww_tiny.txt"
output_file = "/Users/vkodali/IdeaProjects/word_error_rate_whisperweb/data/batch_trans_ww_tiny.csv"

data = []

with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]  # remove blank lines
    for i in range(0, len(lines), 2):  # every pair of lines = filename + text
        audio_file = lines[i].replace("=== ", "").replace(" ===", "")
        transcript = lines[i + 1]
        data.append([audio_file, transcript])

# Write to a CSV file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["audio_file", "transcribed_text"])
    writer.writerows(data)

print(f"✅ CSV created successfully at: {output_file}")

import pandas as pd

# File paths
original_csv = "/Users/vkodali/Downloads/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/overview-of-recordings.csv"
transcribed_csv = "/Users/vkodali/IdeaProjects/word_error_rate_whisperweb/data/batch_trans_ww_tiny.csv"
output_csv = "/Users/vkodali/IdeaProjects/word_error_rate_whisperweb/data/dataset_ww_trans_tiny.csv"

# Load both CSVs
df_original = pd.read_csv(original_csv)
df_transcribed = pd.read_csv(transcribed_csv)

# Merge on file name
merged_df = pd.merge(
    df_original,
    df_transcribed,
    left_on="file_name",
    right_on="audio_file",
    how="inner"
)

# Keep only required columns
merged_df = merged_df[["file_name", "transcribed_text", "phrase"]]

# Save merged CSV
merged_df.to_csv(output_csv, index=False)

print(f"✅ Merged CSV created successfully at: {output_csv}")
print(merged_df.head())


import pandas as pd
import jiwer
from pathlib import Path

# Path to your merged CSV file
csv_path = Path("/Users/vkodali/IdeaProjects/word_error_rate_whisperweb/data/dataset_ww_trans_tiny.csv")

# Load the dataset
df = pd.read_csv(csv_path)

print(f"✅ Loaded {len(df)} rows from: {csv_path}")
print(f"Columns: {list(df.columns)}")

# Create JiWER transformation for normalization
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# Prepare lists to store results
sentence_wers = []
alignment_visualizations = []

# Loop through each row
for i, row in df.iterrows():
    ref = transformation(str(row["phrase"]))
    hyp = transformation(str(row["transcribed_text"]))

    # Compute WER
    wer_score = jiwer.wer(ref, hyp)
    sentence_wers.append(wer_score)

    # Generate alignment visualization
    alignment = jiwer.process_words([ref], [hyp])
    alignment_str = jiwer.visualize_alignment(alignment)
    alignment_visualizations.append(alignment_str)

    print(f"Processed {i+1}/{len(df)} — WER: {wer_score:.4f}")

# Add results to dataframe
df["sentence_wer"] = sentence_wers
df["alignment_visualization"] = alignment_visualizations

# Save the new CSV with alignment info
output_path = Path("/Users/vkodali/IdeaProjects/word_error_rate_whisperweb/data/dataset_ww_trans_tiny_with_wer_alignment.csv")
df.to_csv(output_path, index=False)

# Display summary
average_wer = sum(sentence_wers) / len(sentence_wers)

print("\n=== Word Error Rate (WER) Results ===")
print(f"Average WER: {average_wer:.4f}")
print(f"✅ WER results with alignments saved to: {output_path}")

# Preview a few alignments
print("\n=== Sample Alignment Visualization ===")
print(df["alignment_visualization"].iloc[0])


