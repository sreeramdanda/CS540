from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    dataset = np.load(filename)
    return dataset - np.mean(dataset, axis=0)

def get_covariance(dataset):
    dataset = np.array(dataset)
    return np.dot(1/(len(dataset)-1), np.dot(np.transpose(dataset), np.array(dataset)))

def get_eig(S, m):
    eigenvalues, eigenvectors = eigh(a=S, subset_by_index=[len(S)-m, len(S)-1])
    return np.diag(eigenvalues[::-1]), np.fliplr(eigenvectors)

def get_eig_prop(S, prop):
    requested, eigenvectors = eigh(a=S, subset_by_value=[prop * np.sum(eigh(a=S, eigvals_only=True)), np.inf])
    return np.diag(requested[::-1]), np.fliplr(eigenvectors)

def project_image(image, U):
    return np.dot(U, np.dot(np.transpose(U), image))

def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32), order='F')
    proj = np.reshape(proj, (32, 32), order='F')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.set_title('Original')
    ax2.set_title('Projection')

    pos = ax1.imshow(orig, aspect='equal')
    pos2= ax2.imshow(proj, aspect='equal')

    fig.colorbar(pos, ax=ax1, fraction=0.0453)
    fig.colorbar(pos2, ax=ax2, fraction=0.0453)

    plt.show()