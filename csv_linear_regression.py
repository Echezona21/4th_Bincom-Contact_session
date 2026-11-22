import pandas as pd
#Checking the data
df = pd.read_csv("student_scores.csv")
print(df.head())

X = df[['Hours']]
y = df['Scores']

#Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
#Creating a linear regression model
model = LinearRegression()
#Fitting the model to the training data
model.fit(X_train, y_train)