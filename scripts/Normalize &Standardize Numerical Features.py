import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the cleaned dataset
cleaned_file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\cleaned_data.csv"
df_cleaned = pd.read_csv(cleaned_file_path)

# Identify numerical features
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Normalization: Scale features to a range of [0, 1]
min_max_scaler = MinMaxScaler()
df_normalized = df_cleaned.copy()
df_normalized[numerical_cols] = min_max_scaler.fit_transform(df_cleaned[numerical_cols])

# Standardization: Scale features to have a mean of 0 and a standard deviation of 1
standard_scaler = StandardScaler()
df_standardized = df_cleaned.copy()
df_standardized[numerical_cols] = standard_scaler.fit_transform(df_cleaned[numerical_cols])

# Displaying the first few rows of the normalized and standardized datasets
print("Normalized Data Sample:")
print(df_normalized.head())

print("\nStandardized Data Sample:")
print(df_standardized.head())

# Save the normalized and standardized datasets to new CSV files
normalized_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\normalized_data.csv"
standardized_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\standardized_data.csv"

df_normalized.to_csv(normalized_output_path, index=False)
df_standardized.to_csv(standardized_output_path, index=False)

print(f"Normalized data saved as {normalized_output_path}")
print(f"Standardized data saved as {standardized_output_path}")
