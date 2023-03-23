import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv("C:\student_scores.csv")
print(df.head())

y = df['Scores'].values.reshape(-1, 1)
X = df['Hours'].values.reshape(-1, 1)

print(df['Hours'].values) # [2.5 5.1 3.2 8.5 3.5 1.5 9.2 ... ]
print(df['Hours'].values.shape) # (25,)

print(X.shape) # (25, 1)
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)

print(X_train) # [[2.7] [3.3] [5.1] [3.8] ... ]
print(y_train)


regressor = LinearRegression()

regressor.fit(X_train, y_train)

accuracy = regressor.score(X_test,y_test)
print('Accuracy: ',accuracy)

print('Slpoe: ',regressor.intercept_)
print('Coefficent: ',regressor.coef_)

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# Visualize the test set and the regression line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

def calc(slope, intercept, hours):
    return slope*hours+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)
print(score) # [[94.80663482]]

score = regressor.predict([[9.5]])
print(score) # 94.80663482

y_pred = regressor.predict(X_test)



from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:',mse)
