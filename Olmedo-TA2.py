import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(data.head(10))

#print(data.isnull().sum())

#print(data.describe())



'''
# Pair plot
sns.pairplot(data, hue='species', diag_kind='kde')
plt.show()

# Box plot
for column in iris.feature_names:
    sns.boxplot(x='species', y=column, data=data) 
    plt.title(f' Boxplot of {column}')
    plt.show()

# Histograms
data[iris.feature_names].hist(bins=15, figsize=(10, 6), edgecolor='black') 
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

# Heatmap
correlation_matrix = data.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f') 
plt.title('Feature Correlation Heatmap')
plt.show()

# Violin plot
for column in iris.feature_names:
    sns.violinplot(x='species', y=column, data=data) 
    plt.title(f'Violin Plot of {column}')
    plt.show()

# Swarm plot
for column in iris.feature_names:
    sns.swarmplot(x='species', y=column, data=data) 
    plt.title(f'Swarm Plot of {column}')
    plt.show()
'''


# Features and target
X = data[iris.feature_names] 
y = data['species']

# Standardize the features
scaler = StandardScaler() 
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

# Accuracy and classification report
print(f"Accuracy: {accuracy_score (y_test, y_pred)}") 
print(classification_report(y_test, y_pred))
    

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names) 
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
