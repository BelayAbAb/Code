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
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Histogram for the Amount
sns.histplot(df['Amount'], bins=10, kde=True, ax=axs[0])
axs[0].set_title('Histogram of Transaction Amounts', fontsize=14)
axs[0].set_xlabel('Amount')
axs[0].set_ylabel('Frequency')

# Box Plot for the Amount
sns.boxplot(x=df['Amount'], ax=axs[1])
axs[1].set_title('Box Plot of Transaction Amounts', fontsize=14)
axs[1].set_xlabel('Amount')

# Highlight potential outliers
outliers = df[df['Amount'] > (summary_stats['Mean'] + 3 * summary_stats['Standard Deviation'])]

# Add a scatter plot of outliers
axs[2].scatter(outliers['Amount'], [1]*len(outliers), color='red', label='Potential Outliers')
axs[2].set_title('Potential Outliers in Transaction Amounts', fontsize=14)
axs[2].set_xlabel('Amount')
axs[2].set_yticks([])
axs[2].legend()
axs[2].grid()

# Save the combined plot as a JPG file
output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\distribution_and_outliers.jpg"
plt.tight_layout()
plt.savefig(output_path, format='jpg')
plt.close()

print(f"Combined output saved as {output_path}")
