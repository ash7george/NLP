import pandas as pd

# Load the CSV files (assuming they are in the same directory as this script)
df_roberta = pd.read_csv('Roberta_sentiment_analysis.csv')
df_distillbert = pd.read_csv('distillbert_sentiment_analysis.csv')

# Sort both DataFrames by the 'ticker' column
df_roberta_sorted = df_roberta.sort_values(by='ticker').reset_index(drop=True)
df_distillbert_sorted = df_distillbert.sort_values(by='ticker').reset_index(drop=True)

# Merge the two DataFrames on the 'ticker' and 'summary' columns for direct comparison
df_comparison = pd.merge(df_roberta_sorted, df_distillbert_sorted, on=['ticker', 'summary'],
                         suffixes=('_roberta', '_distillbert'))

# Calculate statistics
comparison_stats = {
    'Mean Sentiment Score - Roberta': df_comparison['sentiment_score_roberta'].mean(),
    'Mean Sentiment Score - DistillBERT': df_comparison['sentiment_score_distillbert'].mean(),
    'Label Agreement Percentage': (df_comparison['sentiment_label_roberta'] == df_comparison['sentiment_label_distillbert']).mean() * 100,
    'Sentiment Score Correlation': df_comparison[['sentiment_score_roberta', 'sentiment_score_distillbert']].corr().iloc[0, 1]
}

# Convert the statistics dictionary to a DataFrame for easier HTML rendering
stats_df = pd.DataFrame(list(comparison_stats.items()), columns=['Metric', 'Value'])

# Save the comparison DataFrame and statistics as an HTML file
html_output = """
<html>
<head><title>Sentiment Analysis Comparison Results</title></head>
<body>
    <h1>Sentiment Analysis Comparison Results</h1>
    <h2>Summary Statistics</h2>
    {stats_table}
    <h2>Sample Comparison Data</h2>
    {comparison_table}
</body>
</html>
"""

# Generate HTML tables for the statistics and sample data
stats_table = stats_df.to_html(index=False)
comparison_table = df_comparison.head().to_html(index=False)

# Write the HTML output to a file
with open('comparison_results.html', 'w') as file:
    file.write(html_output.format(stats_table=stats_table, comparison_table=comparison_table))

print("Results have been saved to 'comparison_results.html'.")
