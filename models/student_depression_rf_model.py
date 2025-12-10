import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load CSV
df = pd.read_csv("../staging/processed_student_depression_dataset.csv")

# Separate features/labels
X = df.drop("depression", axis=1)
y = df["depression"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=300, 
    random_state=42, 
    class_weight="balanced")


print("Training model...")
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 10))
corr = X.corr()  # Correlation of features only
sns.heatmap(corr, cmap="viridis", annot=False)
plt.title("Feature Correlation Matrix")
plt.show()