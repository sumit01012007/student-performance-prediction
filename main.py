# Student Performance Prediction using Machine Learning

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------
# 1. Load Student Performance Dataset
# ---------------------------------------------------------

df = pd.read_csv("D:\\MLOPS-5SEm vishal sir\\project 3\\Student_Performance_DT - Student_Performance_DT.csv")

print("Student Performance Dataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ---------------------------------------------------------
# 2. Select Features and Target
# ---------------------------------------------------------

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Sleep_Hours",
    "Participation",
    "Internet_Access",
    "Extracurricular"
]

target = "Final_Result"

data = df[features + [target]].copy()


# ---------------------------------------------------------
# 3. Convert Categorical Values into Numerical Values
# ---------------------------------------------------------

data["Internet_Access"] = data["Internet_Access"].map({
    "No": 0,
    "Yes": 1
})

data["Extracurricular"] = data["Extracurricular"].map({
    "No": 0,
    "Yes": 1
})

data["Final_Result"] = data["Final_Result"].map({
    "Fail": 0,
    "Pass": 1
})


# ---------------------------------------------------------
# 4. Separate Input and Output
# ---------------------------------------------------------

X = data[features]
y = data[target]


# ---------------------------------------------------------
# 5. Split Dataset into Training and Testing Data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ---------------------------------------------------------
# 6. Create Decision Tree Classifier
# ---------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)


# ---------------------------------------------------------
# 7. Train the Model
# ---------------------------------------------------------

model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")


# ---------------------------------------------------------
# 8. Make Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 9. Calculate Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2f}")


# ---------------------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ---------------------------------------------------------
# 11. Classification Report
# ---------------------------------------------------------

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fail", "Pass"]
))


# ---------------------------------------------------------
# 12. Predict Result for a New Student
# ---------------------------------------------------------

new_student = pd.DataFrame(
    [[6.0, 85.0, 75.0, 8, 7.5, 7, 1, 1]],
    columns=features
)

prediction = model.predict(new_student)


print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("Student is predicted to PASS.")
else:
    print("Student is predicted to FAIL.")