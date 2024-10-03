import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Data Sample:")
print(df.head())

# Evaluate the dataset for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Create a DataFrame to summarize missing data
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

# Print the summary of missing data
print("\nMissing Data Summary:")
print(missing_data_summary[missing_data_summary['Missing Values'] > 0])

# Visualization of missing values with enhanced aesthetics
plt.figure(figsize=(12, 6))
bars = sns.barplot(x=missing_data_summary.index, y='Percentage', data=missing_data_summary,
                   palette='viridis', edgecolor='black')
plt.title('Percentage of Missing Values per Feature', fontsize=18)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage of Missing Values (%)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.axhline(0, color='black', linewidth=0.8)  # Adding a horizontal line at y=0

# Annotate bars with percentage values
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save the missing values bar chart as a JPG file
output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\missing_values_analysis.jpg"
plt.savefig(output_path, format='jpg')
plt.close()

print(f"Missing values analysis saved as {output_path}")

# Determine strategies for handling missing values
print("\nStrategies for Handling Missing Values:")
for feature in missing_data_summary.index:
    if missing_data_summary['Missing Values'][feature] > 0:
        if df[feature].dtype == 'object':
            print(f"For {feature}: Consider using mode imputation.")
        else:
            print(f"For {feature}: Consider using mean or median imputation.")
