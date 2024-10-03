import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the initial summary of missing values
print("Initial Missing Values Summary:")
print(df.isnull().sum())

# Define the threshold for removing columns with excessive missing data
threshold = 0.3  # Example: remove columns with more than 30% missing data
missing_percentage = df.isnull().mean()
columns_to_remove = missing_percentage[missing_percentage > threshold].index

# Drop columns with excessive missing values
df.drop(columns=columns_to_remove, inplace=True)

# Display the DataFrame after dropping columns
print("\nDataFrame after dropping columns with excessive missing values:")
print(df.isnull().sum())

# Imputation using Mean for numerical features
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Imputation using Median for specific features if desired (replace 'specific_column' with actual column name)
# df['specific_column'].fillna(df['specific_column'].median(), inplace=True)

# Alternatively, KNN Imputer can be used for more sophisticated imputation
# knn_imputer = KNNImputer(n_neighbors=5)
# df[num_cols] = knn_imputer.fit_transform(df[num_cols])

# Display the final summary of missing values
print("\nFinal Missing Values Summary:")
print(df.isnull().sum())

# Save the cleaned DataFrame to a new CSV file
cleaned_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\c\cleaned_data.csv"
df.to_csv(cleaned_output_path, index=False)

print(f"Cleaned data saved as {cleaned_output_path}")
