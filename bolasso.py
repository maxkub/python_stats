from sklearn.utils import resample
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_data(n,p,n_var):
    """
    inputs
    n : number of lines in X
    p : number of columns in X, y, theta
    n_var : number of 1 in theta

    output
    X : n*p matrix filled with random values with standard normal distribution
    theta : vector (size=p) with zeros except for the n_var first coordinates
    y : y = X*theta
    """

    X = np.random.randn(n,p)
    theta = [np.random.rand() if i < n_var else 0. for i in range(p)]
    y = np.dot(X,theta)

    return X,y,theta


def bolasso(X,y,m,alpha):

    support = np.ones(np.shape(X)[1])
    for i in range(m):
        Xi,yi = resample(X,y) # creating one bootstrap sample
        lasso = Lasso(alpha=alpha,fit_intercept=True,max_iter=1000)
        lasso.fit(Xi,yi) # applying Lasso on the sample
        support_temp = [ 1. if coef != 0. else 0. for coef in lasso.coef_] # support of the coef_ vector
        support = support_temp * support  # computing support_temp 'inter' support

    columns = [ i for i in range(np.shape(X)[1]) if support[i]==1.]

    X_red = X[:,columns] # selecting the columns in X that are indicated in support
    ols = LinearRegression()
    ols.fit(X_red,y) # ordinary least square using the reduced X matrix

    # building the result of bolasso
    theta = np.zeros(np.shape(X)[1])
    for i in range(len(columns)):
        theta[columns[i]]=ols.coef_[i]

    return theta, ols.intercept_



if __name__ == '__main__' :

    X,y,theta_ = create_data(60,10,6)

    theta, intercept = bolasso(X,y,10,0.2)

    print("theta original " , theta_)
    print("bolasso result " , theta)
