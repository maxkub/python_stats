import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
import scipy as sp
import numpy as np


data1 = pd.read_csv("Galton.txt", sep="\t")

data = data1[['Father', 'Mother', 'Height']]

data[:] = preprocessing.scale(data)

print(data.head())

data['MeanParents'] = 0.5 * (data['Father'] + 1.08 * data['Mother'])


data.plot(kind='scatter', x='MeanParents', y='Height')

regr = linear_model.LinearRegression()

# x = np.asarray(data['MeanParents'])
# y = np.asarray(data['Height'])

x = []
{x.append([i]) for i in data['MeanParents']}
x = np.asarray(x)

y = []
{y.append([i]) for i in data['Height']}
y = np.asarray(y)


#--------------------------------------------------------------
regr.fit(x, y)

theta1 = regr.coef_
theta0 = regr.intercept_

plt.figure()
plt.scatter(x, y, marker='o', cmap='jet')
plt.plot(x, regr.predict(x), 'r', label='fit')
plt.xlabel('MeanParents')
plt.ylabel('Height')
plt.legend(loc='best')


#-----------------------------------------------------------------

plt.figure()
res = []
for i in range(len(x)):
    res.append(y[i][0] - regr.predict(x[i])[0][0])

res = np.asarray(res)

plt.hist(res)
plt.xlabel('residus')


#---------------------------------------------------------------------

regr.fit(y, x)


alpha1 = regr.coef_
alpha0 = regr.intercept_

data_var = data.var()
data_mean = data.mean()

print("mean", data_mean)
print("var", data_var)

alpha0_cal = data_mean['MeanParents'] + data_mean['Height'] / data_mean['MeanParents'] * \
    data_var['MeanParents'] / data_var['Height'] * \
    (theta0 - data_mean['Height'])

print(" alpha0 ", alpha0, "alpha0 calc : ", alpha0_cal)


alpha1_cal = data_var['MeanParents'] / data_var['Height'] * theta1

print(" alpha1 ", alpha1, "alpha1 calc : ", alpha1_cal)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# regression multiple


x = []
{x.append([data.Mother[i], data.Father[i]]) for i in range(len(data))}
x = np.asarray(x)
print("x ", x)

y = []
{y.append([i]) for i in data['Height']}
y = np.asarray(y)


regr = linear_model.LinearRegression()

regr.fit(x, y)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:, 0], x[:, 1], y[:, 0])

X = np.arange(min(data.Mother), max(data.Mother), 2)
Y = np.arange(min(data.Father), max(data.Father), 2)
X, Y = np.meshgrid(X, Y)

Z = []
for i in range(len(X)):
    L = []
    for j in range(len(X[0])):
        L.append(regr.predict([X[i][j], Y[i][j]])[0][0])
    Z.append(L)

print(Z)

ax.plot_wireframe(X, Y, Z, color='r', rstride=1, cstride=1, alpha=0.7)

ax.set_xlabel('Mother')
ax.set_ylabel('Father')


#-------------------------------------------------------------------------

#residu = y - regr.predict(x)

residu = []
for i in range(len(x)):
    residu.append(y[i][0] - regr.predict(x[i])[0][0])

norm2 = 0
for i in residu:
    norm2 += i**2

residu_range = np.arange(min(residu), max(residu))


print('norm2 = ', norm2)


plt.figure()
plt.hist(residu)

kde = sp.stats.gaussian_kde(residu)
plt.plot(residu_range, kde.evaluate(residu_range), 'r')

plt.xlabel('residu 2')


plt.show()
