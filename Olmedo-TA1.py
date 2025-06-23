import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

'''missing_values = df.isnull().sum()

print(df.head())

print("Missing values in each column:\n", missing_values)

duplicate_rows = df.duplicated().sum()
print("Number of duplicate rows: ", duplicate_rows)

df = df.drop_duplicates()
rowDrop = df.duplicated().sum()

print(f"Number of duplicate rows after: {rowDrop}")'''

#scaler = StandardScaler()

#df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

#df = pd.get_dummies(df, columns=['species'], drop_first=True)

#print(df.describe()) 

df['sepal_length'].hist(bins=20)
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')
plt.show()

sns.boxplot(x=df['sepal_length'])
plt.title('Boxplot of Sepal Length')
plt.show()

# I included this line of code to filter numeric columns only
# the Setosa is not included because the corr() method only works with numeric data
numeric_df = df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.scatter(df['sepal_length'], df['petal_length'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Relationship between Sepal Length and Petal Length')
plt.show() 
