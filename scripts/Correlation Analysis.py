import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Data Sample:")
print(df.head())

# Filter out only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Generate a correlation matrix for numerical features
correlation_matrix = numeric_df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()

# Save the heatmap as a JPG file
output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\correlation_matrix.jpg"
plt.savefig(output_path, format='jpg')
plt.close()

print(f"Correlation matrix saved as {output_path}")
