import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("forestfires.csv")
X = dataset.iloc[:, 0:13]
y = dataset.iloc[:, 12].values
X = np.array(X)

labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.scatter(y_test, y_pred, color = 'red')
plt.plot(y_test , y_pred, color = 'blue')
plt.title('Forest Fires')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
