import pandas as pd
import numpy as np

# Load the cleaned dataset
cleaned_file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\cleaned_data.csv"
df_cleaned = pd.read_csv(cleaned_file_path)

# Mock binary target variable for demonstration (1 for good, 0 for bad)
# This is just for testing; adapt it to your actual logic later
df_cleaned['Default'] = np.random.choice([0, 1], size=len(df_cleaned), p=[0.7, 0.3])

# Function to calculate WoE
def calculate_woe(data, target, feature):
    # Create a DataFrame for WoE calculation
    woe_df = data.groupby(feature)[target].agg(['count', 'sum']).reset_index()
    woe_df.columns = [feature, 'Total', 'Good']
    
    # Calculate Bad
    woe_df['Bad'] = woe_df['Total'] - woe_df['Good']
    
    # Calculate proportions
    total_good = woe_df['Good'].sum()
    total_bad = woe_df['Bad'].sum()
    
    # Calculate WoE
    woe_df['Good_Percentage'] = woe_df['Good'] / total_good
    woe_df['Bad_Percentage'] = woe_df['Bad'] / total_bad
    woe_df['WoE'] = np.log(woe_df['Good_Percentage'] / woe_df['Bad_Percentage']).replace([-np.inf, np.inf], 0)

    return woe_df[[feature, 'WoE']]

# Apply WoE Binning to the ProductCategory variable
woe_product_category = calculate_woe(df_cleaned, 'Default', 'ProductCategory')

# Merge WoE values back to the original DataFrame
df_cleaned = df_cleaned.merge(woe_product_category, on='ProductCategory', how='left')

# Rename the WoE column
df_cleaned.rename(columns={'WoE': 'WoE_ProductCategory'}, inplace=True)

# Display the first few rows of the modified DataFrame
print(df_cleaned[['ProductCategory', 'WoE_ProductCategory']].head())

# Save the modified DataFrame with WoE values
woe_output_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\woe_data.csv"
df_cleaned.to_csv(woe_output_path, index=False)

print(f"Woe data saved as {woe_output_path}")
