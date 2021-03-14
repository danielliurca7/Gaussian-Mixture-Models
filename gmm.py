import numpy as np
import os
from random import random, randint
from copy import deepcopy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# the covarience matrix must be non-singular and positive semi-definite
def get_near_psd(A):
    C = (A + A.T)/2
    eigvals, eigvecs = np.linalg.eig(C)
    eigvals = [eigval if eigval > 0 else -eigval + 0.01 for eigval in eigvals]

    return eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)


# calculate the log likelihood
def log_likelihood(D, alpha, miu, cov):
    log_likelihood = 0

    for i in range(len(D)):
        s = 0

        for k in range(K):
            s += alpha[k] * multivariate_normal.pdf(D[i], miu[k], cov[k])

        log_likelihood -= np.log(s)

    return log_likelihood


# rearrange the data in order for the closest distributions to have the same index
def rearrange(ref_alpha, alpha, miu, cov):
    new_alpha = deepcopy(alpha)
    new_miu = deepcopy(miu)
    new_cov = deepcopy(cov)

    for i in range(len(ref_alpha)):
        minim = 1

        for j in range(len(alpha)):
            if abs(ref_alpha[i] - alpha[j]) < minim:
                minim = abs(ref_alpha[i] - alpha[j])

                new_alpha[i] = alpha[j]
                new_miu[i] = miu[j]
                new_cov[i] = cov[j] 

    return new_alpha, new_miu, new_cov


# read the data
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(THIS_FOLDER, 'GMM.in')

D = []

with open(filename, "r") as file:
    for line in file.readlines():
        x = [float(i) for i in line.split() if i.strip()]
        d = len(x)
        D.append(np.array(x))

D = np.array(D)
N = len(D)
K = 4


# initialize reference values
ref_alpha = [0.15, 0.1, 0.5, 0.25]

ref_miu = [
    [0.0, 0.0],
    [5.0, 0.0],
    [-2.0, -5.0],
    [-3.0, 7.0],
]

ref_cov = [
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    [
        [2.0, 1.0],
        [1.0, 2.0],
    ],
    [
        [4.0, -1.3],
        [-1.3, 5.0],
    ],
    [
        [2.7, -1.7],
        [-1.7, 4.2],
    ],
]

ref_log_likelihood = log_likelihood(D, ref_alpha, ref_miu, ref_cov)

# initialize the variables
alpha = np.array([1 / K for _ in range(K)])
miu = np.array([[random() * randint(1, 10) for _ in range(d)] for _ in range(K)])
cov = np.zeros((K, d, d))

for k in range(K):
    for i in range(d):
        for j in range(i+1):
            cov[k][i][j] = randint(1, 10)
            cov[k][j][i] = cov[k][i][j]

    cov[k] = get_near_psd(cov[k])


# EM Algorithm
old_log_likelihood = 10000
new_log_likelihood = 9999
tol = 0.001

while old_log_likelihood - new_log_likelihood > tol:
    old_log_likelihood = new_log_likelihood

    # E-step
    w = np.zeros((N, K))

    for i in range(N):
        x = D[i]

        pdfs = [multivariate_normal.pdf(x, miu[k], cov[k]) for k in range(K)]

        s = sum([alpha[k] * pdfs[k] for k in range(K)])

        for k in range(K):
            w[i][k] = alpha[k] * pdfs[k] / s

    # M-step
    n = np.zeros(K)

    for k in range(K):
        for i in range(N):
            n[k] += w[i][k]

        alpha[k] = n[k] / N    

    for k in range(K):
        miu[k] = np.zeros(d)

        for i in range(N):
            miu[k] += w[i][k] * D[i]

        miu[k] /= n[k]

    for k in range(K):
        cov[k] = np.zeros((d, d))

        for i in range(N):
            cov[k] += w[i][k] * np.outer(D[i] - miu[k], D[i] - miu[k])

        cov[k] /= n[k]

    new_log_likelihood = log_likelihood(D, alpha, miu, cov)

    print(f'{new_log_likelihood} / {ref_log_likelihood}')

alpha, miu, cov = rearrange(ref_alpha, alpha, miu, cov)

print(alpha)
print(miu)
print(cov)


#plot results
colors = ['blue', 'orange', 'green', 'red']

x, y = zip(*D)

assigned_colors = [colors[np.argmax([alpha[k] * multivariate_normal.pdf(x, miu[k], cov[k]) for k in range(K)])] for x in D]
ref_assigned_colors = [colors[np.argmax([ref_alpha[k] * multivariate_normal.pdf(x, ref_miu[k], ref_cov[k]) for k in range(K)])] for x in D]

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Obtained')
ax1.scatter(x, y, marker=".", c=assigned_colors)

ax2.set_title('Reference')
ax2.scatter(x, y, marker=".", c=ref_assigned_colors)

plt.show()