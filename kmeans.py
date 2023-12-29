import copy

import matplotlib.pyplot as plt
import numpy as np
import torch


class ConstrainedSeedKMeans:
    def __init__(self,
                 n_clusters=2,
                 *,
                 n_init=10,
                 max_iter=300,
                 tol=0.0001,
                 verbose=False,
                 invalide_label=-1):
        """Initialization a constrained seed kmeans estimator.

        Args:
            n_clusters: The number of clusters.
            n_init: The number of times the algorithm will run in order to choose
                the best result.
            max_iter: The maximum number of iterations the algorithm will run.
            tol: The convergence threshold of the algorithm. If the norm of a
                matrix, which is the difference between two consective cluster
                centers, is less than this threshold, we think the algorithm converges.
            verbose: Whether to print intermediate results to console.
            invalide_label: Spicial sign to indicate which samples are unlabeled.
                If the y value of a sample equals to this value, then that sample
                is a unlabeled one.
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.INVALID_LABEL = invalide_label

    def _check_params(self, X, y):
        """Check if the parameters of the algorithm and the inputs to it are valid."""
        if type(X) not in (np.ndarray, torch.Tensor):
            raise TypeError(f"Type of X can only take numpy.ndarray and "
                            f"torch.Tensor, but got {type(X)} instead.")

        if type(y) not in (list, np.ndarray, torch.Tensor):
            raise TypeError(f"Type of y can only take list, numpy.ndarray, and"
                            f"torch.Tensor, but got{type(y)} instead.")

        if self.n_clusters > X.shape[0]:
            raise ValueError(f"The number of clusters mube be less than the "
                             f"number of samples.")

        if self.max_iter <= 0:
            raise ValueError(f"The number of maximum iteration must larger than zero.")

    def _init_centroids(self, X, y):
        """Initialize cluster centers with little samples having label."""
        if type(y) == np.ndarray:
            pkg = np
        elif type(y) == torch.Tensor:
            pkg = torch
        elif type(y) == list and type(X) == np.ndarray:
            y = np.array(y)
            pkg = np
        elif type(y) == list and type(X) == torch.Tensor:
            y = torch.Tensor(y)
            pkg = torch
        else:
            raise TypeError('Data type is not supported, please check it again.')

        y_unique = pkg.unique(y)
        if self.INVALID_LABEL in y_unique:
            n_seed_centroids = len(y_unique) - 1
        else:
            n_seed_centroids = len(y_unique)
        assert n_seed_centroids <= self.n_clusters, f"The number of seed centroids" \
                                                    f"should be less than the total" \
                                                    f"number of clusters."

        centers = pkg.empty((self.n_clusters, X.shape[1]), dtype=X.dtype).to(X.device)
        # First, initialize seed centers using samples with label
        for i in range(n_seed_centroids):
            seed_samples = X[y == i]
            centers[i] = seed_samples.mean(axis=0)

        # Then, initilize the remaining centers with random samples from X
        unlabel_idxes = pkg.where(y == self.INVALID_LABEL)[0] # np.where returns a tuple
        for i in range(n_seed_centroids, self.n_clusters):
            idx = np.random.choice(unlabel_idxes, 1, replace=False)
            centers[i] = X[idx]

        return centers, n_seed_centroids

    def _kmeans(self, X, y, init_centers):
        """KMeans algorithm implementation."""
        indices = copy.copy(y)
        if type(indices) == list:
            indices = np.array(indices)
        n_samples, n_features = X.shape[0], X.shape[1]
        cur_centers = init_centers
        new_centers = copy.deepcopy(init_centers)

        # Main loop
        for iter_ in range(self.max_iter):
            # Fist step in KMeans: calculate the closest centroid for each sample
            for i in range(n_samples):
                # If this sample has label, then we use the ground-truth label
                # as its cluster index
                if y[i] != self.INVALID_LABEL:
                    continue

                if type(X) == np.ndarray:
                    min_idx = np.linalg.norm(cur_centers - X[i], axis=1).argmin()
                else:
                    min_idx = torch.norm(cur_centers - X[i], dim=1).argmin()
                indices[i] = min_idx

            # Second step in KMeans: update each centroids
            for i in range(self.n_clusters):
                cluster_samples = X[indices == i]
                # In the case that the cluster is empty, randomly choose
                # a sample from X.
                if cluster_samples.shape[0] == 0:
                    new_centers[i] = X[np.random.choice(n_samples, 1, replace=False)]
                else:
                    new_centers[i] = cluster_samples.mean(axis=0)

            # Calculate inertial at current iteration
            inertia = 0
            for i in range(self.n_clusters):
                if type(X) == np.ndarray:
                    inertia += np.linalg.norm(X[indices == i] - new_centers[i], axis=1).sum()
                else:
                    inertia += torch.norm(X[indices == i] - new_centers[i], dim=1).sum().item()
            if self.verbose:
                print('Iteration {}, inertia: {}'.format(iter_, inertia))

            # Check if KMeans converges
            if type(X) == np.ndarray:
                difference = np.linalg.norm(new_centers - cur_centers, ord='fro')
            else:
                difference = torch.norm(new_centers - cur_centers, p='fro')
            if difference < self.tol:
                if self.verbose:
                    print('Converged at iteration {}.\n'.format(iter_))
                break

            # ATTENSION: Avoid using direct assignment like cur_centers = new_centers
            # This will cause cur_centers and new_cneters to point at the same
            # object in the memory. To fix this, you must create a new object.
            cur_centers = copy.deepcopy(new_centers)

        return new_centers, indices, inertia

    def fit(self, X, y):
        """Using features and little labels to do clustering.

        Args:
            X: numpy.ndarray or torch.Tensor with shape (n_samples, n_features)
            y: List or numpy.ndarray, or torch.Tensor with shape (n_samples,).
                For index i, if y[i] equals to self.INVALID_LABEL, then X[i] is
                an unlabels sample.

        Returns:
            self: The estimator itself.
        """
        self._check_params(X, y)

        _, n_seed_centroids = self._init_centroids(X, y)
        if n_seed_centroids == self.n_clusters:
            self.n_init = 1

        # run constrained seed KMeans n_init times in order to choose the best one
        best_inertia = None
        best_centers, best_indices = None, None
        for i in range(self.n_init):
            init_centers, _ = self._init_centroids(X, y)
            if self.verbose:
                print('Initialization complete')
            new_centers, indices, new_inertia = self._kmeans(X, y, init_centers)
            if best_inertia is None or new_inertia < best_inertia:
                best_inertia = new_inertia
                best_centers = new_centers
                best_indices = indices

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.indices = best_indices

        return self

    def predict(self, X):
        """Predict the associated cluster index of samples.

        Args:
            X: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).

        Returns:
            indices: The associated cluster index of each sample, with shape
            (n_samples,)
        """
        n_samples = X.shape[0]
        indices = [-1 for _ in range(n_samples)]

        for i in range(n_samples):
            if type(X) == np.ndarray:
                min_idx = np.linalg.norm(self.cluster_centers_ - X[i], axis=1).argmin()
            else:
                min_idx = torch.norm(self.cluster_centers_ - X[i], dim=1).argmin()
            indices[i] = min_idx

        if type(X) == np.ndarray:
            return np.array(indices)
        else:
            return torch.tensor(indices)

    def fit_predict(self, X, y):
        """Convenient function."""
        return self.fit(X, y).predict(X)

    def transform(self, X):
        """Transform the input to the centorid space.

        Args:
            X: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).

        Returns:
            output: With shape (n_samples, n_clusters)
        """
        if type(X) == np.ndarray:
            pkg = np
        else:
            pkg = torch

        n_samples = X.shape[0]
        output = pkg.empty((n_samples, self.n_clusters), dtype=X.dtype)
        for i in range(n_samples):
            if type(X) == np.ndarray:
                output[i] = np.linalg.norm(self.cluster_centers_ - X[i], axis=1)
            else:
                output[i] = torch.norm(self.cluster_centers_ - X[i], dim=1)

        return output

    def fit_transform(self, X, y):
        """Convenient function"""
        return self.fit(X, y).transform(X)

    def score(self, X):
        """Opposite of the value of X on the K-means objective."""
        interia = 0
        n_samples = X.shape[0]

        for i in range(n_samples):
            if type(X) == np.ndarray:
                interia += np.linalg.norm(self.cluster_centers_ - X[i], axis=1).min()
            else:
                interia += torch.norm(self.cluster_centers_ - X[i], dim=1).min().item()

        return -1 * interia
