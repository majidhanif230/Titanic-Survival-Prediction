import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(df.head())

# Summary of the dataframe
print("\nSummary of the dataframe:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataframe:")
print(df.isnull().sum())

# Fill missing values for 'Age' and 'Fare' with their median values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
# Fill missing values for 'Embarked' with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert 'Sex' to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numerical values
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Create a new feature 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop irrelevant columns
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId','SibSp','Parch'], inplace=True)

# Display the first few rows after preprocessing
print("\nFirst few rows after preprocessing:")
print(df.head())

# Optional: Data Visualization
print("\nVisualizing survival rate by gender:")
sns.countplot(x='Survived', hue='Sex', data=df)
plt.show()

print("\nVisualizing survival rate by Pclass:")
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.show()

print("\nCorrelation matrix:")
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Split the data into features and target variable
X = df.drop(columns='Survived')
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("Classification Report:")
print(class_report)
