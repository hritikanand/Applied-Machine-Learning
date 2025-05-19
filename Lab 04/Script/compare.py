import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Drop irrelevant columns
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Male=1, Female=0
df['Embarked'].fillna('S', inplace=True)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Select features and target
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Compare kernels using cross-validation
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_scores = {}

print("🔍 Comparing SVM Kernels with 5-Fold Cross-Validation:\n")

for kernel in kernels:
    model = SVC(kernel=kernel)
    scores = cross_val_score(model, X, y, cv=5)
    avg_score = scores.mean()
    kernel_scores[kernel] = avg_score
    print(f"{kernel.capitalize()} Kernel Accuracy: {avg_score:.4f}")

# Find the best kernel
best_kernel = max(kernel_scores, key=kernel_scores.get)
print(f"\n✅ Best kernel: {best_kernel.capitalize()} with accuracy {kernel_scores[best_kernel]:.4f}")

# Optional: Visualize kernel performance
plt.figure(figsize=(8,5))
sns.barplot(x=list(kernel_scores.keys()), y=list(kernel_scores.values()), palette='Set2')
plt.title("SVM Kernel Accuracy Comparison (5-Fold CV)")
plt.xlabel("Kernel Type")
plt.ylabel("Average Accuracy")
plt.ylim(0.6, 0.9)
plt.tight_layout()
plt.show()
