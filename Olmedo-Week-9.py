import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../APENG FILES/Data Analysis/ecommerce_data.csv')


# Check for missing values 
print("Missing values:\n", df.isnull().sum())

# Drop missing values 
df.dropna(inplace=True)


# Convert date column to datetime
df['Date_of_Purchase'] = pd.to_datetime(df['Date_of_Purchase'])

# Filter purchase amounts within a valid range
df = df[(df['Purchase_Amount'] >= 0) & (df['Purchase_Amount'] <= 100000)]
print(df)


mean_purchase = df.groupby('Product_Category')['Purchase_Amount'].mean() 
std_purchase = df.groupby('Product_Category')['Purchase_Amount'].std()

# print(std_purchase)


# Define age groups
def age_group(age):
    if age <= 18:
        return "Teenager"
    elif age <= 35:
        return "Young Adult"
    elif age <= 68:
        return "Adult"
    else:
        return "Senior"
    
# Apply age group function
df['Age_Group'] = df['Age'].apply(age_group)



# Summarize total sales by region
region_sales = df.groupby('Region')['Purchase_Amount'].sum().reset_index()
print(region_sales)


# Line Chart
plt.figure(figsize=(10, 6))
monthly_sales= df.groupby(pd.Grouper (key='Date_of_Purchase', freq='M'))['Purchase_Amount'].sum() 
monthly_sales.plot(title='Total Purchases Over Time')
plt.ylabel('Total Purchase Amount')
plt.show()

# Bar Chart
plt.figure(figsize=(18, 6))
mean_purchase.plot(kind='bar', title='Average Purchase by Product Category')
plt.ylabel('Average Purchase Amount')
plt.show()

# Scatter Plot
plt.figure(figsize=(18, 6))
sns.scatterplot(data=df, x='Age', y='Purchase_Amount', hue='Gender')
plt.title('Age vs Purchase Amount by Gender')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Purchase_Amount'], bins=38, kde=True)
plt.title('Purchase Amount Distribution')
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
correlation = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm') 
plt.title('Correlation Matrix')
plt.show()

# Print key insights
print("Top Performing Regions:\n", region_sales.sort_values(by='Purchase_Amount', ascending=False).head()) 
print("Top Product Categories by Average Purchase:\n", mean_purchase.sort_values (ascending=False).head())
