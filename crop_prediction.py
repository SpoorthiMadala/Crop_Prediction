import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("archive.zip")


X = data.drop(columns=['label'])
y = data['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_grid = {
    'n_estimators': [100, 200],  # Reduced grid
    'max_depth': [None, 10],     # Fewer depth values
    'min_samples_split': [2, 5], # Focus on critical splits
    'max_features': ['sqrt']     # Single feature selection strategy
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_


y_pred_rf = best_rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nOptimized Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")


report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
print("\nClassification Report (Random Forest):")
print(report_rf)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix Heatmap (Random Forest)")
sns.heatmap(conf_matrix_rf, annot=True, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

metrics_df_rf = pd.DataFrame(report_rf).transpose()
metrics_df_rf = metrics_df_rf[['precision', 'recall', 'f1-score']].sort_values(by='f1-score', ascending=False)

metrics_df_rf.plot(kind="bar", figsize=(10, 5))
plt.title('Precision, Recall, and F1-Score by Crop (Random Forest)')
plt.ylabel('Scores')
plt.xlabel('Crops')
plt.show()


xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%")

report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
print("\nClassification Report (XGBoost):")
print(report_xgb)

conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix Heatmap (XGBoost)")
sns.heatmap(conf_matrix_xgb, annot=True, cmap="viridis", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


metrics_df_xgb = pd.DataFrame(report_xgb).transpose()
metrics_df_xgb = metrics_df_xgb[['precision', 'recall', 'f1-score']].sort_values(by='f1-score', ascending=False)

metrics_df_xgb.plot(kind="bar", figsize=(10, 5))
plt.title('Precision, Recall, and F1-Score by Crop (XGBoost)')
plt.ylabel('Scores')
plt.xlabel('Crops')
plt.show()



stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_rf = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=stratified_kfold, n_jobs=-1)
cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=stratified_kfold, n_jobs=-1)

print("\nStratified K-Fold Cross-Validation Accuracy Scores (Random Forest):", cv_scores_rf)
print(f"Mean Accuracy across folds: {cv_scores_rf.mean() * 100:.2f}%")

print("\nStratified K-Fold Cross-Validation Accuracy Scores (XGBoost):", cv_scores_xgb)
print(f"Mean Accuracy across folds: {cv_scores_xgb.mean() * 100:.2f}%")


plt.figure(figsize=(8, 6))
plt.bar(['Random Forest'], cv_scores_rf.mean(), alpha=0.7)
plt.bar(['XGBoost'], cv_scores_xgb.mean(), alpha=0.7)
plt.title('Stratified K-Fold Model Comparison')
plt.ylabel('Average Accuracy')
plt.show()