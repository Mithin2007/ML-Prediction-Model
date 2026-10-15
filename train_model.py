import json
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Small educational dataset:
# hours studied, previous practice score, passed (0 = No, 1 = Yes)
X = [
    [1, 20], [2, 25], [2, 35], [3, 30],
    [3, 45], [4, 40], [4, 50], [5, 45],
    [5, 60], [6, 55], [6, 65], [7, 70],
    [7, 75], [8, 80], [8, 85], [9, 90],
    [1, 15], [2, 20], [3, 35], [4, 45],
    [5, 55], [6, 60], [7, 72], [8, 88],
]
y = [
    0, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1,
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

Path("model").mkdir(exist_ok=True)
with open("model/student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/metrics.json", "w") as f:
    json.dump(
        {"test_accuracy": round(float(accuracy), 4),
         "training_samples": len(X_train),
         "test_samples": len(X_test)},
        f,
        indent=2,
    )

print("Model training complete.")
print(f"Test-set accuracy: {accuracy:.2%}")
print("Saved model to model/student_model.pkl")
print("Saved metrics to model/metrics.json")
