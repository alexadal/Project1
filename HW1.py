import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


x = np.random.rand(100,1)
x = np.sort(x,axis=0)
y = 5*x*x+0.1*np.random.randn(100,1)

#own code for ordinary least squares

#Reshape x

#x = np.sort(x,axis=0)

print(x.shape)
print(x.reshape(100).shape)
X = np.zeros((len(x),3))
print(X)
X[:,0] = 1
X[:,1] = x.reshape(100)
X[:,2] = x.reshape(100)**2




#First order
beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
fit = np.linalg.lstsq(x, y, rcond =None)[0]

ytilde = x@beta
ytilde2 = np.dot(x,fit)




#skicit fit

linreg= LinearRegression()
linreg.fit(x, y)
y_clf = linreg.predict(x)


#Second order

beta_s = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

y_tildes = X@beta_s
linreg2= LinearRegression()
linreg2.fit(X, y)
y_clf2 = linreg2.predict(X)

m_sqr = mean_squared_error(y,y_clf2)
r_score = r2_score(y,y_clf2)


fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.7, lw=2,label='Real fit')
ax.plot(x, ytilde, alpha=0.7, linestyle = 'dashed', lw=2, c='m', label='OLS')
ax.plot(x,ytilde2, alpha=0.7, lw=2,c='k', label = 'OLS')
ax.plot(x,y_clf2, alpha=0.7, linestyle='dashed', lw=2,c='r', label = 'sklearn2')
ax.plot(x,y_tildes, alpha=0.7, lw=2,c='b', label = 'OLS2')
ax.legend()

plt.show()


print("R^2 score", r_score)
print("means squared error",m_sqr)
