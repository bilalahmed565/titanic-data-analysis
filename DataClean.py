import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('train.csv')
print(df.head())
print(df.info())
print(df.isnull().sum().mean())
print(df.describe())
print(df.duplicated().sum())

#Handling Missing Data
df['Age']=df['Age'].fillna(df['Age'].median())
df.drop(columns=['Cabin'], inplace=True) #Droping because 77% data missing it's useless
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum().mean())
df.info()

#EDA
print(df['Survived'].value_counts(normalize=True) * 100) #Overall survival rate
print(df.groupby('Sex')['Survived'].mean()) #Gender VS Survived
print(df.groupby('Pclass')['Survived'].mean()) # Pclass VS Survived
print(df.groupby('Survived')['Age'].mean()) #Age VS Survived

# Saving Clean Data
df.to_csv("titanic_cleaned.csv", index=False)

#Graphs
#Survival Count
sns.countplBot(x='Survived', data=df)
plt.tXitle("Survival Count")
plt.savefig("survival_count.png")
plt.show()

#Gender vs Survival
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Gender vs Survival Rate")
plt.savefig("gender_vs_survival.png")
plt.show()

#Pclass vs Survival
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Passenger Class vs Survival Rate")
plt.savefig("pclass_vs_survival.png")
plt.show()


