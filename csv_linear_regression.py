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

#Making predictions on the test data
# y_prediction = model.predict(int(input("How many hours? ")), X_test)
#print(y_prediction)

y_prediction = model.predict(X_test)
print(y_prediction)

#predicting the score for a student who studies for a certain number of hours
hours = float(input("How many hours studied? "))
predicted_score = model.predict([[hours]])
print("Predicted score:", predicted_score[0])

#Checking the accuracy of the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

#Visualizing the results
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Linear Regression: Hours Studied vs Scores')
plt.show()

