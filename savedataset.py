import pandas as pd
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset('koen430/relevant_selected_stock_news')

# Convert the 'train' split to a pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Save the DataFrame to a CSV file
csv_file_path = 'hugging_face_dataset.csv'
df.to_csv(csv_file_path, index=False)

print(f"Dataset saved as {csv_file_path}")
