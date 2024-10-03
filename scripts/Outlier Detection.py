import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Data Sample:")
print(df.head())

# Filter only numeric columns for outlier detection
numeric_df = df.select_dtypes(include=['number'])

# Set up the figure for box plots
plt.figure(figsize=(15, 10))

# Create box plots for each numeric feature
for i, column in enumerate(numeric_df.columns, start=1):
    plt.subplot(3, 3, i)  # Adjust layout based on the number of numeric features
    sns.boxplot(y=numeric_df[column], palette='viridis')
    plt.title(f'Box Plot of {column}', fontsize=14)
    plt.ylabel(column, fontsize=12)

plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\Week 6\data\box_plots_outliers.jpg", format='jpg')
plt.close()

# Set up the figure for scatter plots
plt.figure(figsize=(15, 10))

# Create scatter plots to visualize relationships and potential outliers
for i, column in enumerate(numeric_df.columns, start=1):
    plt.subplot(3, 3, i)
    sns.scatterplot(data=df, x=numeric_df.columns[0], y=column, alpha=0.6)
    plt.title(f'Scatter Plot: {numeric_df.columns[0]} vs {column}', fontsize=14)
    plt.xlabel(numeric_df.columns[0], fontsize=12)
    plt.ylabel(column, fontsize=12)

plt.tight_layout()
plt.savefig(r"C:\Users\User\Desktop\10Acadamy\Week 6\code\scatter_plots_outliers.jpg", format='jpg')
plt.close()

print("Outlier detection plots saved successfully.")
