import numpy as np
import matplotlib.pyplot as plt

def compute_covariance_matrix(Z):
    """Computes the covariance matrix for a given matrix"""
    return np.matmul((Z-Z.mean()).T, Z-Z.mean())

def find_pcs(cov):
    """Finds the principcal components of a given covariance matrix"""
    vals, vects = np.linalg.eig(cov)
    ind = vals.argsort()[::-1]
    return vects[ind], vals[ind]

def project_data(Z, PCS, L):
    """Data matrix Z, principcal components PCS, eigenvalues L projected into 1 dimension"""
    return np.matmul(PCS[0], Z.T)

def show_plot(Z, Z_star):
    """Shows the original and projected data and saves the plot as an image"""
    plt.title("PCA original and transformed data")
    plt.scatter(Z[:,0], Z[:,1])
    plt.scatter(Z_star, np.zeros(Z_star.shape[0]))
    plt.hlines(0, -20, 20)
    plt.legend(["Original", "Projected"])
    plt.savefig("pca.jpg")