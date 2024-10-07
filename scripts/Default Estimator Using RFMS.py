import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the cleaned dataset
cleaned_file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\cleaned_data.csv"
df_cleaned = pd.read_csv(cleaned_file_path)

# Convert 'TransactionStartTime' to datetime format
df_cleaned['TransactionStartTime'] = pd.to_datetime(df_cleaned['TransactionStartTime'])

# Calculate Recency, Frequency, and Monetary metrics
current_date = df_cleaned['TransactionStartTime'].max()

# Calculate Recency
df_rfm = df_cleaned.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (current_date - x.max()).days,  # Days since last transaction
    'SubscriptionId': 'count',  # Frequency of transactions
    'Amount': 'sum'  # Monetary value
}).reset_index()

df_rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

# Classifying users into good (1) or bad (0) based on RFMS
# Define thresholds (this can be modified based on domain knowledge)
recency_threshold = df_rfm['Recency'].quantile(0.5)
frequency_threshold = df_rfm['Frequency'].quantile(0.5)
monetary_threshold = df_rfm['Monetary'].quantile(0.5)

# Labeling users
df_rfm['RFMS_Score'] = np.where((df_rfm['Recency'] <= recency_threshold) & 
                                 (df_rfm['Frequency'] >= frequency_threshold) & 
                                 (df_rfm['Monetary'] >= monetary_threshold), 1, 0)

# Visualizing RFMS Space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_rfm, x='Monetary', y='Recency', hue='RFMS_Score', palette='coolwarm', alpha=0.7)
plt.axvline(x=monetary_threshold, color='grey', linestyle='--', label='Monetary Threshold')
plt.axhline(y=recency_threshold, color='grey', linestyle='--', label='Recency Threshold')
plt.title('RFMS Space: Recency vs Monetary')
plt.xlabel('Monetary Value (Total Amount)')
plt.ylabel('Recency (Days since last transaction)')
plt.legend()
plt.grid()

# Save the visualization as JPG
rfms_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\rfms_space.jpg"
plt.savefig(rfms_output_path)
plt.close()

print(f"RFMS space visualization saved as {rfms_output_path}")
