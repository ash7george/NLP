import pandas as pd
from transformers import pipeline
import torch  # Import torch to check for GPU availability

# Load the CSV file
input_file = 'hugging_face_dataset2.csv'
df = pd.read_csv(input_file)

# Check if a GPU is available and set the device
if torch.cuda.is_available():
    device = 0  # Use the first GPU
    print("Using GPU for summarization.")
else:
    device = -1  # Use CPU
    print("Using CPU for summarization.")

# Initialize the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# List to store summarized texts
summarized_texts = []

# Set batch size based on your GPU capabilities
batch_size = 10  # Adjust this based on your GPU memory; try 4 or 8 for GTX 1650
max_input_length = 1024  # Maximum input length for BART

# Process the DataFrame in batches
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch = df['text'][start:end].tolist()  # Get a batch of texts

    # Filter out invalid texts and those exceeding max token length
    valid_batch = []
    for text in batch:
        if isinstance(text, str) and len(text.strip()) > 0:
            token_count = len(text.split())  # Simple token count (can also use a more complex tokenizer)
            if token_count <= max_input_length:
                valid_batch.append(text)
            else:
                summarized_texts.append("")  # Leave blank for texts exceeding max token length
                print(f"Text in row {start + batch.index(text) + 1} exceeds max token length and will be skipped.")
        else:
            summarized_texts.append("")  # Leave blank for invalid texts

    if valid_batch:  # Proceed if there are valid texts to summarize
        try:
            # Summarize the texts in the batch
            summaries = summarizer(valid_batch, max_length=130, min_length=30, do_sample=False)

            # Append the summarized texts to the list
            summarized_texts.extend([summary['summary_text'] for summary in summaries])
            print(f"Processed batch {start // batch_size + 1}: {len(summaries)} summaries generated.")
        except Exception as e:
            print(f"Error summarizing batch {start // batch_size + 1}: {e}")
            summarized_texts.extend([""] * (end - start))  # Fill with empty strings for this batch
    else:
        print(f"Batch {start // batch_size + 1}: No valid texts to summarize.")
        summarized_texts.extend([""] * (end - start))  # Fill with empty strings for this batch

# Add summarized texts to the DataFrame (only for the processed rows)
df['summary'] = [""] * len(df)  # Initialize with empty strings
for i in range(len(summarized_texts)):
    df.at[i, 'summary'] = summarized_texts[i]

# Save the results to a new CSV file
output_file = 'summary_results.csv'
df.to_csv(output_file, index=False)

print("Summarization complete. Results saved to", output_file)
