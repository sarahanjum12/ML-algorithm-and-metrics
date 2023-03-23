import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error


# Load the dataset into a pandas dataframe
df = pd.read_csv("C:\Dataset of Diabetes .csv")
print(df.head())


# Preprocess the data

# Drop irrelevant columns such as ID and No. of Patient
df = df.drop(['ID', 'No_Pation'], axis=1)

# Handle missing values

print(df.columns)


# Convert categorical variables such as Gender and Class to numerical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'CLASS'])
print(df.head())
# Split the dataset into training and testing sets
X = df.drop('CLASS_N', axis=1)  # Features
y = df['CLASS_N']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a machine learning algorithm
model = LogisticRegression()

# Train the machine learning algorithm
model.fit(X_train, y_train)
feature_names = X.columns.tolist()
print(feature_names)

print("The evaluation of training data\n")
y_train_pred = model.predict(X_train)
print('Training Accuracy:', accuracy_score(y_train, y_train_pred))
print('Training Precision:', precision_score(y_train, y_train_pred))
print('Training Recall:', recall_score(y_train, y_train_pred))
print('Training F1-score:', f1_score(y_train, y_train_pred))


# Evaluate the performance of the trained model on the testing data
y_pred = model.predict(X_test)
print("The evaluation of testing data/n")
print('Testing Accuracy:', accuracy_score(y_test, y_pred))
print('Testing Precision:', precision_score(y_test, y_pred))
print('Testing Recall:', recall_score(y_test, y_pred))
print('Testing F1-score:', f1_score(y_test, y_pred))

#Evaluate test set mean squared error
mean=mean_squared_error(y_test,y_pred)
print('Mean_Squared_Error: ',mean)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Plot the confusion matrix
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
