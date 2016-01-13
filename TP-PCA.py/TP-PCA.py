import numpy as np
from sklearn.decomposition import PCA
from pandas import read_csv
import matplotlib.pyplot as plt

#%matplotlib inline

colors = np.random.rand(50)
s = 50


def scatter(x, y, lbl):
    plt.figure()
    plt.scatter(x, y, c=colors[:len(lbl)], s=s)
    for xi, yi, lbl in zip(x, y, lbl):
        plt.annotate(lbl, (xi, yi))
    plt.show()

data = read_csv('defra-consumption.csv', delimiter=';', index_col=0)
print(data)

food, countries = data.axes[0], data.axes[1]  # Labels of rows
X = data.values.T  # Matrix sized #examples=countries x #variables=food
Z = X - X.mean(axis=0)  # Center the data
print(Z)

# Q2
pca = PCA(n_components=2)
pca.fit(Z)

Z2 = pca.transform(Z)

print(Z2)

plt.figure()
plt.scatter(Z2[:, 0], Z2[:, 1], color='r', label='PCA')

# Q3

cov = np.cov(Z)
print(cov)

diag, eig_vectors = np.linalg.eig(cov)

norm_vect = eig_vectors * (diag)**0.5 * 4
print(diag)
print(eig_vectors)


print()
print(eig_vectors[:, 0] * 4 * (diag[0])**0.5)
print(eig_vectors[:, 3] * 4 * (diag[3])**0.5)


#S2 = np.dot(norm_vect, Z)
S2 = np.dot(eig_vectors, Z)

S2 = S2[:, 0::3]
plt.scatter(S2[:, 0], S2[:, 1], color='b', label='cov inversion')
#plt.scatter(norm_vect[:, 0], norm_vect[:, 1], color='b', label='cov inversion')


# plt.figure()
#plt.scatter(eig_vectors[:, 0], eig_vectors[:, 1], color='b', label='cov inversion')

# Q4

u, s, v = np.linalg.svd(Z)
print()
# print(v)
print(u)
print(s)

norm_vect = u * (4 * s)**0.5
S2 = np.dot(norm_vect, Z)

S2 = S2[:, 0:2]
plt.scatter(S2[:, 0], S2[:, 1], color='g', label='SVD')

plt.legend(loc='best')

plt.show()
