import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------
data = {
    'Income': [25000, 50000, 75000, 100000, 120000,
               30000, 45000, 80000, 95000, 110000],
    'Age': [25, 35, 45, 50, 23,
            40, 29, 60, 48, 52],
    'Loan_Amount': [20000, 15000, 30000, 25000, 10000,
                    18000, 22000, 35000, 27000, 40000],
    'Credit_Score': ['Poor', 'Fair', 'Good', 'Excellent', 'Poor',
                     'Fair', 'Fair', 'Good', 'Good', 'Excellent']
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Convert Labels to Numbers
# -----------------------------
label_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Excellent': 3
}

df['Credit_Score'] = df['Credit_Score'].map(label_mapping)

# -----------------------------
# Step 3: Split Features & Target
# -----------------------------
X = df[['Income', 'Age', 'Loan_Amount']]
y = df['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -----------------------------
# Step 4: Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=50, random_state=1)
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predict & Evaluate
# -----------------------------
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)

# -----------------------------
# Step 6: Predict New Customer (Correct Format)
# -----------------------------
new_customer = pd.DataFrame(
    [[70000, 40, 20000]],
    columns=['Income', 'Age', 'Loan_Amount']
)

prediction = model.predict(new_customer)

reverse_mapping = {v: k for k, v in label_mapping.items()}

print("\nNew Customer Credit Score:",
      reverse_mapping[prediction[0]])
