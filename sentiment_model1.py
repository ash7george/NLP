# Required libraries installation (run this if not installed)
# !pip install transformers torch pandas scikit-learn plotly datasets

import pandas as pd
import torch
from transformers import pipeline
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import plotly.express as px

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU

# Step 1: Load the CSV file
data = pd.read_csv('summary_results.csv')

# Ensure required columns exist
if 'summary' not in data.columns or 'ticker' not in data.columns:
    raise ValueError("The input file must contain 'summary' and 'ticker' columns.")

# Step 2: Convert the pandas DataFrame to Hugging Face dataset for efficient processing
dataset = Dataset.from_pandas(data[['summary']])

# Step 3: Load pre-trained RoBERTa sentiment model and use GPU if available
sentiment_analyzer = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

# Function to map the sentiment score on a 1-10 scale
def sentiment_to_score(sentiment):
    label = sentiment['label']
    score = sentiment['score']
    if label == 'LABEL_0':  # Negative
        return 1 + (5 - score * 5)
    elif label == 'LABEL_1':  # Neutral
        return 5
    elif label == 'LABEL_2':  # Positive
        return 5 + score * 5

# Step 4: Apply sentiment analysis to the entire dataset in batches
def get_sentiment(batch):
    sentiments = sentiment_analyzer(batch['summary'])
    batch['sentiment'] = sentiments
    batch['sentiment_score'] = [sentiment_to_score(s) for s in sentiments]
    return batch

dataset = dataset.map(get_sentiment, batched=True, batch_size=32)  # Adjust batch size as needed for efficiency

# Convert back to DataFrame for further processing
data['sentiment'] = dataset['sentiment']
data['sentiment_score'] = dataset['sentiment_score']
data['sentiment_label'] = data['sentiment_score'].apply(lambda x: 'positive' if x > 5 else 'negative')

# Step 5: Save the result to a new CSV file
output_file = 'Roberta_sentiment_analysis.csv'
data[['ticker', 'summary', 'sentiment_score', 'sentiment_label']].to_csv(output_file, index=False)

# Step 6: Calculate performance metrics (if ground truth is available)
if 'true_sentiment' in data.columns:
    true_labels = data['true_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    predicted_labels = data['sentiment_label'].apply(lambda x: 1 if x == 'positive' else 0)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

    # Save statistics to CSV
    stats = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    stats.to_csv('sentiment_analysis_statistics.csv', index=False)

# Step 7: Visualize sentiment scores in HTML with color gradient
fig = px.scatter(data, x='ticker', y='sentiment_score',
                 color='sentiment_score', color_continuous_scale='RdYlGn',
                 title="Sentiment Score Visualization")

# Save the plot to an HTML file
fig.write_html("sentiment_analysis_visualization.html")

print("Sentiment analysis complete, results saved to CSV and HTML visualization created.")
