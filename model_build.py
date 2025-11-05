# %%
import pandas as pd
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report,make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV

# %%
data = pd.read_csv('C:/Users/venka/Downloads/Employee-Attrition.csv')

# %%
data

# %%
data.head()

# %%
data.dtypes

# %%
data.shape

# %%
data.isnull().sum()

# %%
data.columns

# %%
for i in data.columns:
    print(f"{i}: {data[i].unique()}")

# %%
columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df = data.drop(columns=columns_to_drop)

# %%
df.columns.to_list()

# %%
df.select_dtypes(include=['object']).columns.to_list()

# %%
df.select_dtypes(include=['int64']).columns.to_list()

# %%
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# %%
df.head()

# %%
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Overall Attrition Count
plt.figure(figsize=(6, 4))
attrition_counts = df['Attrition'].value_counts().sort_index()
attrition_counts.index = ['Stayed', 'Left']
sns.barplot(x=attrition_counts.index, y=attrition_counts.values, palette='viridis')
plt.title('Overall Employee Attrition Count')
plt.xlabel('Attrition')
plt.ylabel('Number of Employees')

plt.show()

# 2. Attrition by Marital Status
plt.figure(figsize=(8, 5))
attrition_by_marital = df.groupby('MaritalStatus')['Attrition'].mean().sort_values(ascending=False)
sns.barplot(x=attrition_by_marital.index, y=attrition_by_marital.values, palette='pastel')
plt.title('Attrition Rate by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=0)
plt.show()

# 3. Attrition by Overtime
plt.figure(figsize=(7, 5))
attrition_by_overtime = df.groupby('OverTime')['Attrition'].mean().sort_values(ascending=False)
sns.barplot(x=attrition_by_overtime.index, y=attrition_by_overtime.values, palette='coolwarm')
plt.title('Attrition Rate for Employees Working Overtime')
plt.xlabel('OverTime')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=0)
plt.show()

# 4. Attrition by Job Satisfaction
plt.figure(figsize=(8, 5))
attrition_by_jobsat = df.groupby('JobSatisfaction')['Attrition'].mean().sort_values(ascending=False)
sns.barplot(x=attrition_by_jobsat.index, y=attrition_by_jobsat.values, palette='magma')
plt.title('Attrition Rate by Job Satisfaction')
plt.xlabel('Job Satisfaction (1=Low, 4=High)')
plt.ylabel('Attrition Rate')
plt.xticks(rotation=0)
plt.show()

# 5. Age Distribution of Employees Who Left vs. Stayed
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Attrition'] == 1]['Age'], label='Left', fill=True, color='red', alpha=0.5)
sns.kdeplot(df[df['Attrition'] == 0]['Age'], label='Stayed', fill=True, color='green', alpha=0.5)
plt.title('Age Distribution of Attrited vs. Non-Attrited Employees')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Attrition')
plt.show()

# %%
df.dtypes

# %%
object_data=df.select_dtypes(include=['object'])

# %%
numeric_data=df.select_dtypes(include=['int64'])

# %%
import plotly.express as px

# %%
df_long = df.melt(value_vars=numeric_data.columns, var_name='Variable', value_name='Value')

# %%
fig = px.box(df_long, x="Variable", y="Value", title='Outliers in Numerical Columns (Interactive)')

# %%
fig.write_html("interactive_outlier_boxplots.html")

# %%
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# %%
num_cols = X.select_dtypes(include='int64').columns
cat_cols = X.select_dtypes(include='object').columns

# %%
smote_sampler = SMOTE(random_state=42)

# %%
numerical_pipeline = Pipeline(steps=[
    ('log_transform', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
])

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num_features', numerical_pipeline, num_cols),
        ('cat_features', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ])

# %%
pipeline_attrition = ImbPipeline(steps=[('preprocessor', preprocessor),
                            ('smote', smote_sampler),
                            ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# %%
pipeline_attrition.fit(X_train, y_train)

# %%
with open('attrition_model.pkl', 'wb') as f:
    pickle.dump(pipeline_attrition, f)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# %%
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

# %%
grid_search = GridSearchCV(pipeline_attrition, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# %%
grid_search.fit(X_train, y_train)

# %%
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation ROC-AUC score: {:.4f}".format(grid_search.best_score_))
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test set ROC-AUC score: {:.4f}".format(roc_auc_score(y_test, y_pred)))

y_pred_train = best_model.predict(X_train)
print("Training set ROC-AUC score: {:.4f}".format(roc_auc_score(y_train, y_pred_train)))

# %%
y_train_pred = pipeline_attrition.predict(X_train)
y_train_proba = pipeline_attrition.predict_proba(X_train)[:, 1]

print("\n--- Model Evaluation on TRAINING Data (with SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"F1-Score: {f1_score(y_train, y_train_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_train, y_train_proba):.4f}")
print("\nClassification Report (Training):\n", classification_report(y_train, y_train_pred))
print("\nConfusion Matrix (Training):\n", confusion_matrix(y_train, y_train_pred))

print("-" * 50)

y_test_pred = pipeline_attrition.predict(X_test)
y_test_proba = pipeline_attrition.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation on TESTING Data (with SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")
print("\nClassification Report (Testing):\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix (Testing):\n", confusion_matrix(y_test, y_test_pred))

# %%

def winsorize_all_columns(X):
    for column in X.columns:
        lower_bound = X[column].quantile(0.05)
        upper_bound = X[column].quantile(0.95)
        X[column] = np.clip(X[column], lower_bound, upper_bound)
    return X

df = pd.read_csv('C:/Users/venka/Downloads/Employee-Attrition.csv')

columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber', 'Attrition']
df_cleaned = df.drop(columns=columns_to_drop)

X = df_cleaned.drop(['PerformanceRating', 'PercentSalaryHike'], axis=1)
y = df_cleaned['PerformanceRating']

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

pipeline_performance = ImbPipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num_features', Pipeline(steps=[
                ('winsorize', FunctionTransformer(winsorize_all_columns)),
                ('scaler', StandardScaler())]), numerical_cols),
            ('cat_features', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)])),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

scoring = make_scorer(f1_score, average='weighted', zero_division=0)

print("Starting hyperparameter tuning with GridSearchCV for Logistic Regression...")
grid_search = GridSearchCV(pipeline_performance, param_grid, cv=5, scoring=scoring, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Final Model Evaluation on TESTING Data (after Tuning) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open('performance_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


