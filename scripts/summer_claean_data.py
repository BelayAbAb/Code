import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
cleaned_file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\cleaned_data.csv"
df_cleaned = pd.read_csv(cleaned_file_path)

# Display descriptive statistics for numerical variables
numerical_summary = df_cleaned.describe()
print("Numerical Features Summary:")
print(numerical_summary)

# Summarizing categorical variables
categorical_summary = df_cleaned.select_dtypes(include=['object']).describe()
print("\nCategorical Features Summary:")
print(categorical_summary)

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Plot histograms for numerical features
plt.subplot(2, 2, 1)
df_cleaned.hist(bins=30, figsize=(15, 10), layout=(3, 2), edgecolor='black')
plt.suptitle('Histograms of Numerical Features')

# Plot bar charts for categorical features
plt.subplot(2, 2, 2)
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
for i, column in enumerate(categorical_columns):
    plt.subplot(2, 2, i+3)
    sns.countplot(y=column, data=df_cleaned, palette='viridis')
    plt.title(f'Count of {column}')
    plt.xlabel('Count')

# Adjust layout
plt.tight_layout()

# Save summary figures as a JPG file
summary_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\summary_patterns.jpg"
plt.savefig(summary_output_path)
plt.close()

print(f"Summary patterns saved as {summary_output_path}")
