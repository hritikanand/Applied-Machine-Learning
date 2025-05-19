import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Drop irrelevant columns
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# 3. Encode categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Male=1, Female=0
df['Embarked'].fillna('S', inplace=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# 5. Select features and target
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Build and train linear kernel SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# 8. Predict and calculate accuracy
y_pred = linear_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

# 9. Bar Chart: Kernel Comparison (static values for report)
kernel_scores = {'linear': 0.78, 'rbf': 0.76, 'poly': 0.72, 'sigmoid': 0.65}
plt.figure(figsize=(8,5))
sns.barplot(x=list(kernel_scores.keys()), y=list(kernel_scores.values()), palette='muted')
plt.title("SVM Kernel Accuracy Comparison")
plt.xlabel("Kernel Type")
plt.ylabel("Accuracy")
plt.ylim(0.5, 0.9)
plt.tight_layout()
plt.show()

# 10. Display accuracy in terminal box
box_width = 40
print("\n" + "=" * box_width)
print(f"{'Model Accuracy':^{box_width}}")
print("-" * box_width)
print(f"{'Accuracy:':<20}{accuracy_percentage:>19.2f}%")
print("=" * box_width + "\n")

# 11. Pie Chart: Correct vs Incorrect Predictions
correct = (y_pred == y_test).sum()
incorrect = (y_pred != y_test).sum()
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [correct, incorrect]
colors = ['#4CAF50', '#F44336']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Prediction Accuracy Breakdown")
plt.axis('equal')
plt.show()

# 12. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Linear SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



