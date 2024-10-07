# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    ])

# Step 5: Define models with increased max_iter
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),  # Increased max_iter
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Hyperparameter tuning parameters
param_grid = {
    'Logistic Regression': {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg']
    },
    'Random Forest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
}

# Step 6: Train and evaluate each model
results = {}

for model_name, model in models.items():
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])
    
    # Perform hyperparameter tuning
    if model_name == 'Logistic Regression':
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='f1', verbose=1)
        grid_search.fit(X_train, y_train)
    else:
        grid_search = RandomizedSearchCV(pipeline, param_grid[model_name], n_iter=10, cv=5, scoring='f1', random_state=42)
        grid_search.fit(X_train, y_train)
    
    # Store the best model and its score
    best_model = grid_search.best_estimator_
    results[model_name] = {
        'best_model': best_model,
        'best_score': grid_search.best_score_,
    }

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    results[model_name]['accuracy'] = accuracy_score(y_test, y_pred)
    results[model_name]['precision'] = precision_score(y_test, y_pred)
    results[model_name]['recall'] = recall_score(y_test, y_pred)
    results[model_name]['f1_score'] = f1_score(y_test, y_pred)
    results[model_name]['roc_auc'] = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    # Step 7: Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
    
    # Save confusion matrix plot
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f"C:/Users/User/Desktop/10Acadamy/Week 6/code/cm_{model_name}.jpg")
    plt.show()

# Step 8: Prepare data for plotting
metric_names = ['Best CV Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
metric_values = {model: [results[model]['best_score'],
                         results[model]['accuracy'],
                         results[model]['precision'],
                         results[model]['recall'],
                         results[model]['f1_score'],
                         results[model]['roc_auc']] for model in models.keys()}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.15  # Width of the bars
x = np.arange(len(metric_names))  # the label locations

for i, model in enumerate(models.keys()):
    ax.bar(x + i * width, metric_values[model], width, label=model)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Training and Evaluation Metrics')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(metric_names)
ax.legend()

# Save the plot as JPG in the specified local folder
output_plot_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\model_evaluation_metrics.jpg"
plt.tight_layout()
plt.savefig(output_plot_path)
plt.show()

# Save the results to a CSV file in the specified local folder
output_results_path = r"C:\Users\User\Desktop\10Acadamy\Week 6\code\model_results.csv"
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv(output_results_path)

print("Model evaluation metrics and results saved successfully.")
