import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Original Data Sample:")
print(df.head())

# Identify categorical variables
nominal_features = ['ProductCategory', 'ChannelId']
# Assume 'PricingStrategy' is an ordinal variable; you can modify this based on your understanding of the data
ordinal_feature = 'PricingStrategy'

# One-Hot Encoding for nominal variables
df_one_hot = pd.get_dummies(df, columns=nominal_features, drop_first=True)

# Label Encoding for ordinal variables
label_encoder = LabelEncoder()
df_one_hot[ordinal_feature] = label_encoder.fit_transform(df_one_hot[ordinal_feature])

# Display the first few rows of the encoded dataframe
print("\nEncoded Data Sample:")
print(df_one_hot.head())

# Save the encoded dataframe to a new CSV file
encoded_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\encoded_data.csv"
df_one_hot.to_csv(encoded_output_path, index=False)

print(f"Encoded data saved as {encoded_output_path}")
