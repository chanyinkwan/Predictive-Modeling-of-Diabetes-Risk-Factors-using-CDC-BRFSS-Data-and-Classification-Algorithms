import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# Configure plot styles
sns.set_style('whitegrid')
# Use a higher resolution setting for better output quality for portfolio
plt.rcParams['figure.dpi'] = 150 
plt.rcParams['figure.figsize'] = [12, 8]
print("Setup complete. Libraries imported successfully.")



FIGURE_COUNT = 0
# Define an output directory for cleanliness, or use './' for the current directory
OUTPUT_PATH = './' 

def save_figure(fig_description):
    """Saves the current matplotlib figure to a PNG file."""
    global FIGURE_COUNT
    FIGURE_COUNT += 1
    filename = f"figure_{FIGURE_COUNT}_{fig_description.replace(' ', '_').lower()}.png"
    filepath = os.path.join(OUTPUT_PATH, filename)
    
    # Use plt.gcf() to get the current figure object
    plt.gcf().savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close() # Close the figure to free up memory
    print(f"Figure saved: {filepath}")


# **ACTION REQUIRED: Replace 'YOUR_DATA_FILE_NAME.csv' with your actual file name**
# Assuming the file is in a 'data' folder one level up.
try:
    # Use the filename provided in your file upload snippet for consistency
    data = pd.read_csv('Data Set/diabetes_012_health_indicators_BRFSS2015_modified2.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("ERROR: Data file not found. Please verify the file path and name.")
    print("Ensure your BRFSS 2015 data is saved as a CSV in the correct directory.")
    # Create dummy data for demonstration if file is missing
    data = pd.DataFrame(np.random.rand(100, 22), 
                        columns=['Diabetes_012', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
                                 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'])

print(f"Dataset Shape: {data.shape}")
print("\nFirst 5 rows of the dataset:")
print(data.head())



# Feature and Target Separation
# Treating as Binary Classification (0: No Diabetes, 1: Diabetes/Pre-diabetes) 
# based on the class distribution found in the output.
TARGET = 'Diabetes_012'
FEATURES = [col for col in data.columns if col != TARGET]

X = data[FEATURES]
y = data[TARGET].astype(int) 

print(f"\nTarget Class Distribution Before Oversampling:\n{y.value_counts()}")


# Perdforming EDA
print("\n--- Generating EDA Plots ---")
variables = {
    'HighBP': 'categorical',
    'HighChol': 'categorical',
    'BMI': 'continuous',
    'GenHlth': 'continuous',
    'Age': 'continuous',
    'Income': 'continuous'
}
custom_palette = sns.color_palette("viridis", 2) # Use 2 colors for 2 classes (0 and 1)
class_labels = ['No Diabetes (0)', 'Diabetes/Pre-diabetes (1)']


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15))
axes = axes.flatten()

for i, (var, var_type) in enumerate(variables.items()):
    if var in data.columns:
        if var_type == 'categorical':
            # Bar plot (using sum for binary variables)
            plot_data = data.groupby(TARGET)[var].sum().reset_index(name='Count')
            sns.barplot(x=TARGET, y='Count', data=plot_data, ax=axes[i], palette=custom_palette, hue=TARGET, legend=False)
            axes[i].set_title(f'Total Count of Positive Responses for {var} by Diabetes Status', fontsize=14)
            axes[i].set_ylabel(f'Sum of {var} (Positive Response)', fontsize=12)
        elif var_type == 'continuous':
            # Box plot for continuous variables
            sns.boxplot(x=TARGET, y=var, data=data, ax=axes[i], palette=custom_palette, hue=TARGET, legend=False)
            axes[i].set_title(f'Box Plot of {var} by Diabetes Status', fontsize=14)
            axes[i].set_ylabel(var, fontsize=12)
        
        axes[i].set_xlabel('Diabetes Status', fontsize=12)
        axes[i].set_xticks([0, 1]) 
        axes[i].set_xticklabels(class_labels)

plt.tight_layout()
plt.suptitle('Exploratory Data Analysis: Key Indicators vs. Diabetes Status', y=1.02, fontsize=16)
save_figure('eda_distributions') # Save Figure 1


# Visualization 2: Correlation Heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', 
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of All Variables')
save_figure('correlation_heatmap') # Save Figure 2


# Modeling Preparation

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nData split into Training and Testing sets.")

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)

print(f"\nTarget Class Distribution After SMOTE:\n{Counter(y_oversampled)}")



# Model Training
print("\n--- Training Random Forest Model ---")

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_oversampled, y_oversampled)

print("Random Forest Classifier trained successfully.")


# Model Evaluation

# Predictions on the Test Set
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test) 

# Calculate metrics for binary classification
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
# Binary ROC-AUC (using probability of the positive class [:, 1])
roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1]) 


print("\n--- Random Forest Model Evaluation Metrics (Test Set) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")
print(f"ROC-AUC (Binary): {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
# disp.plot() implicitly uses plt.figure(), so we can save it after plotting
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest Classifier')
save_figure('confusion_matrix') # Save Figure 3


# Feature Importance Plot
feature_importance = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
# Plot only top 15 features for clarity
feature_importance.head(15).plot(kind='barh', color='darkgreen')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()
save_figure('feature_importance') # Save Figure 4

print("\n--- Top 15 Feature Importances ---")
print(feature_importance.head(15))