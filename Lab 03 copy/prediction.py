import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Drop irrelevant columns
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Male = 1, Female = 0
df['Embarked'].fillna('S', inplace=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # Create Embarked_Q, Embarked_S

# Handle missing Age values
df['Age'].fillna(df['Age'].median(), inplace=True)

# 1. Feature selection and train-test split
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 2. Predictions and evaluation
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 3. Display theta (coefficients)
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Theta (Coefficient)': log_reg.coef_[0]
})
print("\nTheta (Coefficient) Values:")
print(coeff_df)

# Coefficient Visualization
plt.figure(figsize=(10,6))
sns.barplot(data=coeff_df, x='Theta (Coefficient)', y='Feature')
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 4. Prediction on 3 custom passengers
custom_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Age': [29, 35, 45],
    'SibSp': [0, 1, 1],
    'Parch': [0, 0, 2],
    'Fare': [100, 7.25, 21],
    'Sex': [0, 1, 1],  # 0 = female, 1 = male
    'Embarked_Q': [0, 0, 1],
    'Embarked_S': [1, 1, 0]
})


predictions = log_reg.predict(custom_passengers)
labels = ['survived' if pred == 1 else 'not survived' for pred in predictions]
for i, result in enumerate(labels):
    print(f"Passenger {i+1}: {result}")

# 5. Try different split and max_iter
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg_alt = LogisticRegression(max_iter=2000)
log_reg_alt.fit(X_train_alt, y_train_alt)

# Evaluating alternative configuration
y_pred_alt = log_reg_alt.predict(X_test_alt)
acc_alt = accuracy_score(y_test_alt, y_pred_alt)
recall_alt = recall_score(y_test_alt, y_pred_alt)
print("\nAfter changing test size and max_iter:")
print(f"New Accuracy: {acc_alt:.2f}")
print(f"New Recall: {recall_alt:.2f}")

# Adding this after your test prediction
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (survived)

# Plottting probability distribution
plt.figure(figsize=(8, 5))
sns.histplot(y_prob, bins=30, kde=True, color='green')
plt.title("Distribution of Predicted Survival Probabilities")
plt.xlabel("Predicted Probability of Survival")
plt.ylabel("Frequency")
plt.show()
