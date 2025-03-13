import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




with open("digit_embeddings.pkl", 'rb') as f:
    data = pickle.load(f)
filenames = []
Y = []
X = []
for f,y,x in data:
    filenames.append(f)
    Y.append(y)
    X.append(x)
X = np.array(X)
Y = np.array(Y)

# Split Data (Train: 70%, Eval: 15%, Test: 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)}, Eval size: {len(X_eval)}, Test size: {len(X_test)}")

# Move data to GPU using DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
deval = xgb.DMatrix(X_eval, label=y_eval, nthread=-1)
dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)

# Define model parameters for GPU
params = {
    "objective": "multi:softmax",
    "num_class": 10,
    "eval_metric": "mlogloss",
    "tree_method": "hist",  # GPU-accelerated
    "device": "cuda",  # Ensure model and data are on GPU
    "verbosity": 0
}

# Train XGBoost Model
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(deval, "eval")], early_stopping_rounds=10)
model.save_model("xgboost_digit_classifier.model")

model = xgb.XGBClassifier()
model.load_model("xgboost_digit_classifier.model")
# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 11), yticklabels=range(1, 11))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the trained model
model.save_model("xgboost_digit_classifier.model")
print("Model saved as xgboost_digit_classifier.model")