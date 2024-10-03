import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Data Sample:")
print(df.head())

# Calculate Summary Statistics for numerical features
summary_stats = {
    'Mean': df['Amount'].mean(),
    'Median': df['Amount'].median(),
    'Standard Deviation': df['Amount'].std(),
    'Min': df['Amount'].min(),
    'Max': df['Amount'].max(),
    'Range': df['Amount'].max() - df['Amount'].min()
}

# Print Summary Statistics
print("\nSummary Statistics for 'Amount':")
print(summary_stats)

# Create a single figure for the matrix layout
fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})

# Plot Summary Statistics as a table
axs[0].axis('tight')
axs[0].axis('off')
table_data = [[key, value] for key, value in summary_stats.items()]
table = axs[0].table(cellText=table_data, colLabels=['Statistic', 'Value'], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
axs[0].set_title('Summary Statistics for Transaction Amounts', fontsize=14)

# Plot the Distribution of Amount
sns.histplot(df['Amount'], bins=10, kde=True, ax=axs[1])
axs[1].set_title('Distribution of Transaction Amounts', fontsize=14)
axs[1].set_xlabel('Amount')
axs[1].set_ylabel('Frequency')

# Add mean and median lines
axs[1].axvline(summary_stats['Mean'], color='red', linestyle='dashed', linewidth=1, label='Mean')
axs[1].axvline(summary_stats['Median'], color='blue', linestyle='dashed', linewidth=1, label='Median')
axs[1].legend()
axs[1].grid()

# Save the combined plot as a JPG file
output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\summary_statistics_and_histogram.jpg"
plt.tight_layout()
plt.savefig(output_path, format='jpg')
plt.close()

print(f"Combined output saved as {output_path}")
