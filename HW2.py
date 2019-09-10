import numpy as np
import sklearn.linear_model as skl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


x = np.random.rand(100,1)
x = np.sort(x,axis=0)
y = 5*x*x+0.1*np.random.randn(100,1)

#Set lamdba

l_i = 0.02
l_i2 = 0.8
l_i3 = 0.1

print(x.shape)
print(x.reshape(100).shape)
X = np.zeros((len(x),3))
print(X)
X[:,0] = 1
X[:,1] = x.reshape(100)
X[:,2] = x.reshape(100)**2

l_vec = np.eye(len(X[0]))*l_i

#Skicit

clf_ridge =  skl.Ridge(alpha=l_i2).fit(X, y)

clf_lasso = skl.Lasso(alpha=l_i3).fit(X,y)




print(l_vec)
beta_ridge = np.linalg.inv(X.T.dot(X)+l_vec).dot(X.T).dot(y)


y_ridge = X@beta_ridge
y_clf_ridge = clf_ridge.predict(X)
y_clf_lasso = clf_lasso.predict(X)

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.7, lw=2,label='Real fit')
ax.plot(x, y_ridge, alpha=0.7, linestyle = 'dashed', lw=2, c='m', label='Ridge')
ax.plot(x, y_clf_ridge, alpha=0.7, linestyle = 'dashed', lw=2, c='r', label='Ridge Sk')
ax.plot(x, y_clf_lasso, alpha=0.7, linestyle = 'dashed', lw=2, c='b', label='Lasso Sk')


ax.legend()

plt.show()


"""
Part 3
"""


beta_rs = clf_ridge
beta_os = skl.LinearRegression().fit(X,y)
beta_ls = clf_lasso


print(beta_rs.coef_)
print(beta_os.coef_)

var_br = np.var(beta_rs.coef_)
var_os = np.var(beta_os.coef_)
var_bl = np.var(beta_ls.coef_)

print("----------------------------------")

print("Var Beta Ridge", var_br)
print("Var Beta OLS", var_os)
print("Var Beta Lasso", var_bl)


"""
Get Statistical properties
"""

#MSE

print("Ridge MSE",mean_squared_error(y,y_clf_ridge))
print("Lasso MSE",mean_squared_error(y,y_clf_lasso))

#R2

print("Ridge R-Score",r2_score(y,y_clf_ridge))
print("Lasso R-Score",r2_score(y,y_clf_lasso))
