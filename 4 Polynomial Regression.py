# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#Split (no need)
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)"""

#Feature Scaling (no need, will be implemented automaticly)
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)""

#fiting linear regression to dataset
#indeed no need to fit linear regression but we want to compare
#linear regression results with polynomial reg resulsts
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
y_pred=lin_reg.predict(X)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
#till here we just transformed our X to X_poly
#now we really fitted our polynomial regression 
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly, y)
y_pred_2=lin_reg2.predict(X_poly)


￼
#visualising linear regression results
%matplotlib inline
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Truth or Bluf (Linear Regression)')
plt.xlabel=('position level')
plt.ylabel=('salary')
plt.show()

#visualising polynomial regression results
plt.scatter(X, y, color='red')
plt.plot(X, y_pred_2, color='blue')
plt.title('Truth or Bluf (Polynomial Regression)')
plt.xlabel=('position level')
plt.ylabel=('salary')
plt.show()


