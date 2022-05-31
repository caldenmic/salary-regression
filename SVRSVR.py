#SVR

#REGRESSION TEMPLATE

#data preprocessing template

#importing libreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values    # 1:2 because it becomes a matrix        
Y = dataset.iloc[:, 2].values      # 2 because it becomes a vector    

#taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])"""

#encoding catagorical data
"""
#encoding independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#encoding dependent variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#AVOIDING THe DUMMY VARIABLE TRAP
X = X[:,1:]"""

#Splitting the dataset into training set and test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1, 1))            #important to reshape cuz you get 1D array instead of 2D


#FITTING SVR MODEL TO THE DATASET
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

#PREDICTING A NEW RESULT WITH SVR MODEL
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#SVR RESULTS
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('REGRESSION MODEL')
plt.xlabel('POSITION LEVEL')
plt.ylabel('SALARY')
plt.show()

#SVR RESULTS (FOR HIGHER RESOLUTION AND SMOOTHER CURVE)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('REGRESSION MODEL')
plt.xlabel('POSITION LEVEL')
plt.ylabel('SALARY')
plt.show()





