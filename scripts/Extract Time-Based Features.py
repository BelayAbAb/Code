import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Convert the 'TransactionStartTime' column to datetime format
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Derive new features from the 'TransactionStartTime'
df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
df['Transaction_Day'] = df['TransactionStartTime'].dt.day
df['Transaction_Month'] = df['TransactionStartTime'].dt.month
df['Transaction_Year'] = df['TransactionStartTime'].dt.year

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Plotting transaction counts by hour
plt.subplot(2, 2, 1)
sns.countplot(x='Transaction_Hour', data=df, palette='viridis')
plt.title('Transaction Counts by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Transaction Count')

# Plotting transaction counts by day
plt.subplot(2, 2, 2)
sns.countplot(x='Transaction_Day', data=df, palette='magma')
plt.title('Transaction Counts by Day')
plt.xlabel('Day of Month')
plt.ylabel('Transaction Count')

# Plotting transaction counts by month
plt.subplot(2, 2, 3)
sns.countplot(x='Transaction_Month', data=df, palette='cividis')
plt.title('Transaction Counts by Month')
plt.xlabel('Month of Year')
plt.ylabel('Transaction Count')

# Plotting transaction counts by year
plt.subplot(2, 2, 4)
sns.countplot(x='Transaction_Year', data=df, palette='crest')
plt.title('Transaction Counts by Year')
plt.xlabel('Year')
plt.ylabel('Transaction Count')

# Adjust layout
plt.tight_layout()

# Save the figure as a JPG file
pattern_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\temporal_patterns_summary.jpg"
plt.savefig(pattern_output_path)
plt.close()

print(f"Temporal features patterns summary saved as {pattern_output_path}")
