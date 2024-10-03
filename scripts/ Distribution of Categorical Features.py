import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Data Sample:")
print(df.head())

# Analyze categorical features
categorical_features = ['ProductCategory', 'ChannelId']

# Create a single figure for the matrix layout
fig, axs = plt.subplots(len(categorical_features), 1, figsize=(10, 10))

for i, feature in enumerate(categorical_features):
    # Bar plot for categorical feature
    sns.countplot(data=df, x=feature, ax=axs[i], order=df[feature].value_counts().index)
    axs[i].set_title(f'Frequency Distribution of {feature}', fontsize=14)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Frequency')
    axs[i].tick_params(axis='x', rotation=45)  # Rotate x labels for better visibility

# Save the combined plot as a JPG file
output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\categorical_feature_distribution.jpg"
plt.tight_layout()
plt.savefig(output_path, format='jpg')
plt.close()

print(f"Combined output saved as {output_path}")
