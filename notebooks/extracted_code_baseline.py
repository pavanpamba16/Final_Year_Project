# ==========================================
# CODE CELL 1
# ==========================================
#!pip install catboost
#!pip install lime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import missingno as msno
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    cohen_kappa_score, f1_score, roc_curve, roc_auc_score, auc,
    accuracy_score, classification_report, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, precision_score,
    recall_score
)
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm

import lime
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_tabular


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold



# ==========================================
# CODE CELL 2
# ==========================================
#   first read the data file
df= pd.read_csv('/content/Malaria-Data.csv')

# ==========================================
# CODE CELL 3
# ==========================================
df.shape

# ==========================================
# CODE CELL 4
# ==========================================
df.head()

# ==========================================
# CODE CELL 5
# ==========================================
df.describe()

# ==========================================
# CODE CELL 6
# ==========================================
df.info()

# ==========================================
# CODE CELL 7
# ==========================================
df.columns

# ==========================================
# CODE CELL 8
# ==========================================
df.isnull().sum()

# ==========================================
# CODE CELL 9
# ==========================================



# Compute Pearson correlation coefficients
corr_matrix = df.corr()

# Extract correlations of features with the target variable
target_corr = corr_matrix['severe_maleria']

# Plot heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(target_corr.to_frame(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation of Features with severe_malaria ')
plt.xlabel('Features')
plt.ylabel('Target Variable')
plt.show()


# ==========================================
# CODE CELL 10
# ==========================================
X = df.drop(columns=['severe_maleria'])  # Features
y = df['severe_maleria']  # Target variable

# Calculate the correlation coefficients between features and the target variable
correlation_scores = X.corrwith(y)

# Select features with correlation coefficient above a threshold (e.g., 0.01)
selected_features = correlation_scores[abs(correlation_scores) > 0.01].index.tolist()
# Include the target variable in the selected features
selected_features.append('severe_maleria')

# Create a new DataFrame with selected features and target variable
df_selected = df[selected_features]
# Print the selected features
print("Selected features:", selected_features)


# ==========================================
# CODE CELL 11
# ==========================================
df_copy = df_selected.copy()

# ==========================================
# CODE CELL 12
# ==========================================
correlation_scores

# ==========================================
# CODE CELL 13
# ==========================================
# Create a pie chart of the distribution of Malaria result

# Define labels and colors
custom_colors = ['#B43757', '#a37b85']
labels = ['Non-Severe Malaria', 'Severe Malaria']

#get the counts
result_counts = df_copy['severe_maleria'].value_counts()

# Create pie chart
plt.pie(result_counts, labels=labels, autopct='%1.1f%%', colors=custom_colors, shadow=True)
plt.title('Distribution of Malaria Result')
plt.show()


# ==========================================
# CODE CELL 14
# ==========================================
# Create a count plot of the Malaria result

ax = sns.countplot(x='severe_maleria', data=df_copy, palette=custom_colors)
plt.title('Count of Malaria Result')

# Add labels to the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 0.1, '{:.0f}'.format(height), ha='center')

# Remove spines
sns.despine(left=True, bottom=True)

plt.show()


# ==========================================
# CODE CELL 15
# ==========================================
# Create the figure and subplots
fig, axs = plt.subplots(11, 4, figsize=(15, 30))

# Flatten the axs array to make it easier to iterate through
axs = axs.flatten()

# Loop through each feature and plot a histogram in a separate subplot
for i, col in enumerate(df_copy.columns):
    axs[i].hist(df_copy[col], density = True, bins=10)
    axs[i].set_title(col)

# Hide any unused subplots
for i in range(len(df_copy.columns), 11 * 4):
    axs[i].axis('off')

# Add a main title and adjust the spacing between subplots
#plt.suptitle('Histograms for Multiple Features')
plt.subplots_adjust(hspace=0.6, wspace=0.6)
#fig.suptitle("Histogram for the features", fontsize = 30)

# Show the plot
plt.show()

# ==========================================
# CODE CELL 16
# ==========================================
# Create the correlation matrix
correlation_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(15, 9))

# Plot the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Add title
plt.title('Correlation Matrix')

# Show plot
plt.show()


# ==========================================
# CODE CELL 17
# ==========================================
# Calculate the correlation matrix
corr_matrix = df_copy.corr().unstack().sort_values().drop_duplicates()
# Display the correlation matrix
print(corr_matrix)

# ==========================================
# CODE CELL 18
# ==========================================
X = df_copy.drop(["severe_maleria"], axis=1) #independent
y = df_copy.severe_maleria #dependent

# ==========================================
# CODE CELL 19
# ==========================================
# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=101)

# ==========================================
# CODE CELL 20
# ==========================================
# Quick sanity check with the shapes of Training and testing datasets
print("X_train - ",X_train.shape)
print("y_train - ",y_train.shape)
print("X_test - ",X_test.shape)
print("y_test - ",y_test.shape)

# ==========================================
# CODE CELL 21
# ==========================================
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# CODE CELL 22
# ==========================================
X_test

# ==========================================
# CODE CELL 23
# ==========================================
import xgboost
print(xgboost.__version__)


# ==========================================
# CODE CELL 24
# ==========================================
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 1.0],
    },
    "Gradient Boost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "gamma": [0, 0.1, 0.2],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [4, 6, 8],
    },
}


# ==========================================
# CODE CELL 25
# ==========================================
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)


# ==========================================
# CODE CELL 26
# ==========================================
pip install -U scikit-learn xgboost


# ==========================================
# CODE CELL 27
# ==========================================
Store_models = [
    ("Random Forest", RandomForestClassifier(random_state=123)),
    ("AdaBoost", AdaBoostClassifier(random_state=123)),
    ("Gradient Boost", GradientBoostingClassifier(random_state=123)),
    ("XGBoost", XGBClassifier(random_state=123)),
    ("CatBoost", CatBoostClassifier(random_state=123, silent=True)),
]


# ==========================================
# CODE CELL 28
# ==========================================
# Initialize an empty list to store results
tuned_results_list = []

for clf_name, clf in Store_models:
    print(f"Tuning hyperparameters for {clf_name}...")

    # Get the hyperparameter grid for the current model
    param_grid = param_grids[clf_name]

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=20,  # Number of different combinations to try
        cv=cv_strategy,
        scoring='accuracy',  # Metric to optimize for
        n_jobs=-1,  # Use all available processors
        verbose=1,
        random_state=123
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best estimator
    best_model = random_search.best_estimator_

    # Make predictions with the best model
    predictions = best_model.predict(X_test)
    ypred_prob = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    score = accuracy_score(y_test, predictions)
    rocAuc_score = roc_auc_score(y_test, ypred_prob)
    mcc = matthews_corrcoef(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    kappa = cohen_kappa_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Store the results
    results_dict = {
        "Model": clf_name,
        "Best Params": random_search.best_params_,  # Store best params
        "Accuracy Score": score,
        'Roc_Auc_score': rocAuc_score,
        "MCC": mcc,
        "Balanced_Acc": bal_acc,
        "Cohen_Kappa": kappa,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    # Append the dictionary to the list
    tuned_results_list.append(results_dict)

# Convert the list of dictionaries to a DataFrame
Tuned_model_results = pd.DataFrame(tuned_results_list)

# Sort by MCC or any other metric of interest
Tuned_model_results.sort_values(by="MCC", ascending=False)


# ==========================================
# CODE CELL 29
# ==========================================
# Drop the "Best Params" column from the final DataFrame
Tuned_model_results.drop(columns=["Best Params"], inplace=True)

# Sort by MCC or any other metric of interest
Tuned_model_results.sort_values(by="MCC", ascending=False)


# ==========================================
# CODE CELL 30
# ==========================================


# ==========================================
# CODE CELL 31
# ==========================================
for model_name, model in Store_models:
    # Fit the model with your training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix using seaborn's heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


# ==========================================
# CODE CELL 32
# ==========================================
y_test_series = y_test.squeeze()  # Convert DataFrame to Series
# Get the value counts of the Series
value_counts = y_test_series.value_counts()
print(value_counts)


# ==========================================
# CODE CELL 33
# ==========================================
# Plot ROC curves for each classifier
plt.figure(figsize=(8, 6))

for clf_name, clf in Store_models:
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classifiers')
plt.legend(loc='lower right')
plt.show()

# ==========================================
# CODE CELL 34
# ==========================================
# over sampling of the dataset to get a balanced dataset
class_0 = df_copy[df_copy['severe_maleria'] == 0]
class_1 = df_copy[df_copy['severe_maleria'] == 1]

# over sampling of the minority class 1
class_1_over = class_1.sample(len(class_0), replace=True)

# Creating a new dataframe with over sampled class 1 df and class 0 df
df_new = pd.concat([class_1_over, class_0], axis=0)

# plotting the new label distribution
df_new['severe_maleria'].value_counts().plot(kind='bar', title='Label Distribution after Oversampling')
plt.ylabel('Count')
plt.show()

# ==========================================
# CODE CELL 35
# ==========================================


# over sampling of the dataset to get a balanced dataset
class_0 = df_copy[df_copy['severe_maleria'] == 0]
class_1 = df_copy[df_copy['severe_maleria'] == 1]

# over sampling of the minority class 1
class_1_over = class_1.sample(len(class_0), replace=True)

# Creating a new dataframe with over sampled class 1 df and class 0 df
df_new = pd.concat([class_1_over, class_0], axis=0)

# plotting the new label distribution
df_new['severe_maleria'].value_counts().plot(kind='bar', title='Label Distribution after Oversampling')
plt.ylabel('Count')
plt.show()

X = df_new.drop('severe_maleria', axis=1)  # features
y = df_new[['severe_maleria']]  # labels

# splitting the features and labels into train and test with test size = 20% and train size = 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Define the models
Store_models_over = [
    ("Random Forest", RandomForestClassifier(random_state=123)),
    ("AdaBoost", AdaBoostClassifier(random_state=123)),
    ("Gradient Boost", GradientBoostingClassifier(random_state=123)),
    ("XGBoost", XGBClassifier(random_state=123)),
    ("CatBoost", CatBoostClassifier(random_state=123, silent=True)),
]

# Define hyperparameter grids
param_grids_over = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 1.0],
    },
    "Gradient Boost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "gamma": [0, 0.1, 0.2],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [4, 6, 8],
    },
}

# Initialize an empty list to store results
results_list_over = []

for clf_name, clf in tqdm(Store_models_over):
    print(f"Tuning hyperparameters for {clf_name} using RandomizedSearchCV...")

    # Get the hyperparameter grid for the current model
    param_grid = param_grids_over[clf_name]

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=10,  # Number of random combinations to try (adjust based on time)
        cv=5,  # Cross-validation
        scoring='accuracy',  # Metric to optimize for
        n_jobs=-1,  # Use all available processors
        verbose=1,
        random_state=123
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train.values.ravel())

    # Get the best estimator
    best_model = random_search.best_estimator_

    # Make predictions with the best model
    predictions = best_model.predict(X_test)
    ypred_prob = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    score = accuracy_score(y_test, predictions)
    rocAuc_score = roc_auc_score(y_test, ypred_prob)
    mcc = matthews_corrcoef(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    kappa = cohen_kappa_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    # Store results in a dictionary
    results_dict = {
        "Model": clf_name,
        "Accuracy Score": score,
        'Roc_Auc_score': rocAuc_score,
        "MCC": mcc,
        "Balanced_Acc": bal_acc,
        "Cohen_Kappa": kappa,
        "Precision": precision,
        "Recall": recall
    }

    # Append the dictionary to the list
    results_list_over.append(results_dict)

# Convert the list of dictionaries to a DataFrame
Store_model_results_over = pd.DataFrame(results_list_over)

# Sort by Precision or any other metric
Store_model_results_over.sort_values(by="Precision", ascending=False)

# Display the results
print(Store_model_results_over)


# ==========================================
# CODE CELL 36
# ==========================================
Store_model_results_over

# ==========================================
# CODE CELL 37
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# Sort the accuracy table
Accuracy_table = Store_model_results_over[["Model", "Accuracy Score"]].sort_values(by="Accuracy Score", ascending=False)

# Create the barplot
fig, ax = plt.subplots(figsize=(12, 11))
sns.barplot(y='Model', x='Accuracy Score', data=Accuracy_table, ax=ax).set(title="Accuracy Score")

# Add labels to the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}',
                ((p.get_x() + p.get_width()), p.get_y()),
                xytext=(5, -10),
                textcoords='offset points')

plt.show()

# ==========================================
# CODE CELL 38
# ==========================================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

for model_name, model in Store_models_over:
    # Fit the model with your training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix using seaborn's heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


# ==========================================
# CODE CELL 39
# ==========================================
y_test_series = y_test.squeeze()  # Convert DataFrame to Series
# Get the value counts of the Series
value_counts = y_test_series.value_counts()
print(value_counts)


# ==========================================
# CODE CELL 40
# ==========================================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Plot ROC curves for each classifier
plt.figure(figsize=(8, 6))

for clf_name, clf in Store_models_over:
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classifiers')
plt.legend(loc='lower right')
plt.show()

# ==========================================
# CODE CELL 41
# ==========================================
# Initialize RandomForestClassifier
random_forest = RandomForestClassifier(random_state=123)

# Train the model
random_forest.fit(X_train, y_train)

# Get feature importances from the trained model
importances = random_forest.feature_importances_

# Get the indices of the top 10 features with the highest importance scores
indices = np.argsort(importances)[-10:]

# Plot the top 10 features with their importance scores in a bar chart
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importance Score')
plt.title('Top 10 Features in RandomForestClassifier')
plt.show()


# ==========================================
# CODE CELL 42
# ==========================================
# Initialize CatBoostClassifier
catboost = CatBoostClassifier(random_state=123, silent=True)

# Train the model
catboost.fit(X_train, y_train)

# Get feature importances from the trained model
importances = catboost.get_feature_importance()

# Get the indices of the top 10 features with the highest importance scores
indices = np.argsort(importances)[-10:]

# Plot the top 10 features with their importance scores in a bar chart
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Importance Score')
plt.title('Top 10 Features in CatBoostClassifier')
plt.show()


# ==========================================
# CODE CELL 43
# ==========================================
Accuracy_table = Store_model_results_over[["Model","Accuracy Score"]].sort_values(by = "Accuracy Score", ascending = False)
Accuracy_table

# ==========================================
# CODE CELL 44
# ==========================================
fig, ax = plt.subplots(figsize=(12, 10))

# drawing the plot
sns.barplot(y='Model', x='Accuracy Score', data=Accuracy_table, ax=ax).set(title="Accuracy Score")

# Add labels to the bars
for i in ax.patches:
    ax.text(i.get_width(), i.get_y() + 0.5 * i.get_height(), f'{i.get_width():.4f}', fontsize=10, color='black')

plt.show()


# ==========================================
# CODE CELL 45
# ==========================================
MCC_table = Store_model_results_over[["Model", "MCC"]].sort_values(by="MCC", ascending=False)
fig, ax = plt.subplots(figsize=(14, 9))

# drawing the plot
sns.barplot(x='Model', y='MCC', data=MCC_table, ax=ax)
plt.title("Matthew's Correlation Coefficient", fontweight='bold', fontsize=15)

# Add labels to the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.3f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 10),
                   textcoords = 'offset points')

plt.show()


# ==========================================
# CODE CELL 46
# ==========================================

instance_to_explain = X_test.iloc[0]

# Define your feature names with labels
feature_names_with_labels = list(X_train.columns)
feature_names_with_labels.append("Severe Malaria")

# Create a LIME explainer for tabular data with feature names including labels
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="classification", feature_names=feature_names_with_labels, class_names=['No Malaria', 'Severe Malaria'])

# Explain the model's prediction for the chosen instance
explanation = explainer.explain_instance(instance_to_explain.values, random_forest.predict_proba, num_features=10)

# Display the explanation
explanation.show_in_notebook()


# ==========================================
# CODE CELL 47
# ==========================================

instance_to_explain = X_test.iloc[0]

# Define your feature names with labels
feature_names_with_labels = list(X_train.columns)
feature_names_with_labels.append("Severe Malaria")

# Create a LIME explainer for tabular data with feature names including labels
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="classification", feature_names=feature_names_with_labels, class_names=['No Malaria', 'Severe Malaria'])

# Explain the model's prediction for the chosen instance
explanation = explainer.explain_instance(instance_to_explain.values, catboost.predict_proba, num_features=10)

# Display the explanation
explanation.show_in_notebook()


# ==========================================
# CODE CELL 48
# ==========================================
clf = CatBoostClassifier(random_state=70)

# Train the model
clf.fit(X_train, y_train, verbose=False)
#clf.fit(X_train_resampled, y_train_resampled, verbose=False)



# Evaluate the model's accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy of the model:", accuracy)

# Function to calculate permutation feature importance
def permutation_importance(model, X, y, metric, num_iterations=100):
    baseline_score = metric(y, model.predict(X))
    importances = {}
    for col in X.columns:
        importance_scores = []
        for _ in range(num_iterations):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            permuted_score = metric(y, model.predict(X_permuted))
            #importance_scores.append(permuted_score - baseline_score)
            importance_scores.append(baseline_score - permuted_score)
        importances[col] = np.mean(importance_scores)
    return importances

# Calculate permutation feature importance
importances = permutation_importance(clf, X_test, y_test, accuracy_score)

# Sort feature importances
sorted_importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=False)}

# Print feature importances
print("\nPermutation Feature Importances:")
for feature, importance in sorted_importances.items():
    print(f"{feature}: {importance:.4f}")

# Plotting feature importances
plt.figure(figsize=(10, 6))

# Assigning colors based on values
colors = ['blue' if v >= 0 else 'red' for v in sorted_importances.values()]

plt.barh(range(len(sorted_importances)), list(sorted_importances.values()), color=colors, align='center')
plt.yticks(range(len(sorted_importances)), list(sorted_importances.keys()))
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importances CatBoost')
plt.show()


# ==========================================
# CODE CELL 49
# ==========================================
#Train a random forest classifier
clf = RandomForestClassifier(random_state=120)
# clf.fit(X_train, y_train)


# Train the model
#clf.fit(X_train, y_train)
clf.fit(X_train, y_train)



# Evaluate the model's accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy of the model:", accuracy)

# Function to calculate permutation feature importance
def permutation_importance(model, X, y, metric, num_iterations=100):
    baseline_score = metric(y, model.predict(X))
    importances = {}
    for col in X.columns:
        importance_scores = []
        for _ in range(num_iterations):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            permuted_score = metric(y, model.predict(X_permuted))
            #importance_scores.append(permuted_score - baseline_score)
            importance_scores.append(baseline_score - permuted_score)
        importances[col] = np.mean(importance_scores)
    return importances

# Calculate permutation feature importance
importances = permutation_importance(clf, X_test, y_test, accuracy_score)

# Sort feature importances
sorted_importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=False)}

# Print feature importances
print("\nPermutation Feature Importances:")
for feature, importance in sorted_importances.items():
    print(f"{feature}: {importance:.4f}")

# Plotting feature importances
plt.figure(figsize=(10, 6))

# Assigning colors based on values
colors = ['blue' if v >= 0 else 'red' for v in sorted_importances.values()]

plt.barh(range(len(sorted_importances)), list(sorted_importances.values()), color=colors, align='center')
plt.yticks(range(len(sorted_importances)), list(sorted_importances.keys()))
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importances Random Forest')
plt.show()


# ==========================================
# CODE CELL 50
# ==========================================


# ==========================================
# CODE CELL 51
# ==========================================
!pip install shap


# ==========================================
# CODE CELL 52
# ==========================================
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Fit the Random Forest model
best_rf_model = RandomForestClassifier(random_state=123, n_estimators=300)
best_rf_model.fit(X_train, y_train)

# Create a SHAP explainer object
explainer = shap.TreeExplainer(best_rf_model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Ensure we're working with the correct SHAP values for the positive class
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]  # For the positive class
else:
    shap_values_positive = shap_values

# If the shape is (n_samples, n_features, n_classes), flatten the third dimension
if shap_values_positive.ndim == 3:
    shap_values_positive = shap_values_positive[:, :, 1]  # Get SHAP values for the positive class

# Create an Explanation object
expl = shap.Explanation(values=shap_values_positive, data=X_test)

# Beeswarm plot using the Explanation object
shap.plots.beeswarm(expl)  # Pass the Explanation object



# ==========================================
# CODE CELL 53
# ==========================================



# Calculate absolute mean SHAP values
mean_shap_values = np.abs(shap_values_positive).mean(axis=0)

# Create a DataFrame for easier manipulation
feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Mean SHAP Value': mean_shap_values})

# Sort values by mean SHAP value
feature_importance = feature_importance.sort_values(by='Mean SHAP Value', ascending=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Mean SHAP Value'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.title('Mean Absolute SHAP Values for Features')
plt.show()


# ==========================================
# CODE CELL 54
# ==========================================


