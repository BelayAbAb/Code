# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning  # Added import for ConvergenceWarning
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Step 1: Load the cleaned dataset from a local path
cleaned_file_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\cleaned_data.csv"
df_cleaned = pd.read_csv(cleaned_file_path)

# Check for missing values and data types
print("Data types:\n", df_cleaned.dtypes)
print("Missing values:\n", df_cleaned.isnull().sum())

# Step 2: Mock binary target variable for demonstration (1 for good, 0 for bad)
df_cleaned['FraudResult'] = np.random.choice([0, 1], size=len(df_cleaned), p=[0.7, 0.3])

# Define features and target
X = df_cleaned.drop(columns=['FraudResult', 'CustomerId', 'SubscriptionId', 'TransactionId'])  # Exclude non-predictive columns
y = df_cleaned['FraudResult']

# Step 3: Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Define preprocessing steps for numerical and categorical features
numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 5: Define the Random Forest model
model = RandomForestClassifier()

# Hyperparameter tuning parameters for Random Forest
param_dist = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

# Step 6: Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)])

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=5, scoring='f1', random_state=42, verbose=1)
random_search.fit(X_train, y_train)

# Store the best model and its score
best_model = random_search.best_estimator_
best_score = random_search.best_score_

# Evaluate on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Step 7: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])

# Save confusion matrix plot
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest')
plt.savefig(r"C:/Users/User/Desktop/10Acadamy/Week 6/code/cm_Random_Forest.jpg")
plt.show()

# Step 8: Prepare data for plotting
metric_names = ['Best CV Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
metric_values = [best_score, accuracy, precision, recall, f1, roc_auc]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4  # Width of the bars
x = np.arange(len(metric_names))  # the label locations

ax.bar(x, metric_values, width, label='Random Forest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Random Forest Model Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)

# Save the plot as JPG in the specified local folder
output_plot_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\random_forest_evaluation_metrics.jpg"
plt.tight_layout()
plt.savefig(output_plot_path)
plt.show()

# Save the results to a CSV file in the specified local folder
output_results_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\random_forest_results.csv"
results_df = pd.DataFrame({
    'Metric': metric_names,
    'Score': metric_values
})
results_df.to_csv(output_results_path, index=False)

print("Random Forest model evaluation metrics and results saved successfully.")
