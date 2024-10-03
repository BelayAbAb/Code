import pandas as pd

# Load the dataset from the specified CSV file path
file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\data\data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Data Sample:")
print(df.head())

# Group by CustomerId to calculate aggregate features
aggregate_features = df.groupby('CustomerId').agg(
    Total_Transaction_Amount=('Amount', 'sum'),
    Average_Transaction_Amount=('Amount', 'mean'),
    Transaction_Count=('TransactionId', 'count'),
    Std_Dev_Transaction_Amount=('Amount', 'std')
).reset_index()

# Display the aggregate features
print("\nAggregate Features:")
print(aggregate_features.head())

# Save the aggregate features to a new CSV file
aggregate_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\aggregate_features.csv"
aggregate_features.to_csv(aggregate_output_path, index=False)

print(f"Aggregate features saved as {aggregate_output_path}")
