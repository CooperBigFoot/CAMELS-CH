import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sktime.clustering.k_means import TimeSeriesKMeans
import seaborn as sns
from scipy.stats import zscore
from typing import List, Dict, Union


RANDOM_SEED = 42


class TimeSeriesClusterer:
    def __init__(
        self,
        n_clusters: int = 15,  # Default to 15 clusters as in paper
        metric: str = "dtw",  # Dynamic Time Warping
        warping_window: float = 0.2,  # 20% warping window as mentioned in paper
        n_init: int = 5,
        max_iter: int = 75,
    ):
        """
        Initialize the Time Series Clusterer with enhanced parameters.

        Args:
            n_clusters (int): Number of clusters to create
            metric (str): Distance metric for clustering (default: DTW)
            warping_window (float): Size of warping window for DTW (proportion of series length)
            n_init (int): Number of times the algorithm runs with different centroid seeds
            max_iter (int): Maximum number of iterations for clustering
        """
        self.n_clusters = n_clusters
        self.warping_window = warping_window

        # Create distance parameters for DTW with LB_Keogh
        distance_params = {
            "window": warping_window,  # Set the warping window constraint
            "use_lb": True,  # Enable LB_Keogh lower bounding
            "lb_method": "keogh",  # Specify LB_Keogh as the lower bounding method
        }

        self.clusterer = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            random_state=RANDOM_SEED,
            distance_params=distance_params,
            averaging_method="dba",  # Use Dynamic Time Warping Barycenter Averaging
        )

        self.id_to_index = {}
        self.labels_ = None
        self.X = None
        self.series_ids = None
        self.cluster_centers_ = None

        self.optimization_results: Dict[str, List[Union[int, float]]] = {
            "n_clusters": [],
            "inertia": [],
            "silhouette_scores": [],
        }

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess time series data by standardizing (z-score normalization)

        Args:
            X (np.ndarray): Input time series data

        Returns:
            np.ndarray: Standardized time series data
        """
        return zscore(X, axis=1)

    def fit(self, X: np.ndarray, series_ids: List[str]) -> "TimeSeriesClusterer":
        """
        Fit the clusterer to the data

        Args:
            X (np.ndarray): Input time series data
            series_ids (List[str]): Unique identifiers for each time series

        Returns:
            TimeSeriesClusterer: Fitted clusterer
        """
        # Preprocess data
        processed_X = self.preprocess_data(X)

        # Store the input data
        self.X = processed_X
        self.series_ids = series_ids

        # Map IDs to indices
        self.id_to_index = {id_: idx for idx, id_ in enumerate(series_ids)}

        # Fit the clusterer
        self.clusterer.fit(processed_X)
        self.labels_ = self.clusterer.labels_
        self.cluster_centers_ = self.clusterer.cluster_centers_

        return self

    def get_label_from_id(self, series_id: str) -> int:
        """
        Get cluster label for a specific series ID

        Args:
            series_id (str): Unique identifier of the time series

        Returns:
            int: Cluster label
        """
        if series_id not in self.id_to_index:
            raise ValueError(f"Series ID {series_id} not found")
        idx = self.id_to_index[series_id]
        return self.labels_[idx]

    def plot_clusters(self, max_series_per_cluster: int = 10):
        """
        Plot clusters with centroids and sample series from each cluster in a grid layout.

        Args:
            max_series_per_cluster (int): Maximum number of series to plot per cluster
        """
        # Define distinct colors for centroids
        centroid_colors = sns.color_palette("husl", self.n_clusters)

        # Calculate grid dimensions
        cols = min(3, self.n_clusters)  # Max 3 columns
        rows = (self.n_clusters + cols - 1) // cols  # Ceiling division to get rows

        # Create a single figure with subplots
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 5, rows * 4), squeeze=False
        )

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Loop through each cluster
        for i in range(self.n_clusters):
            ax = axes[i]

            # Plot series in this cluster
            cluster_series = self.X[self.labels_ == i]

            # Limit number of series plotted
            cluster_series = cluster_series[:max_series_per_cluster]

            # Plot individual series in gray
            for series in cluster_series:
                ax.plot(series, color="gray", alpha=0.4)

            # Plot centroid in color
            ax.plot(
                self.cluster_centers_[i, 0],
                color=centroid_colors[i % len(centroid_colors)],
                linewidth=3,
                label="Cluster Centroid",
            )

            ax.set_title(f"Cluster {i}")
            ax.set_xlabel("Week")
            ax.set_ylabel("Standardized Flow")
            ax.legend()
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

        # Hide unused subplots
        for j in range(self.n_clusters, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def describe_clusters(self) -> List[dict]:
        """
        Provide descriptive statistics for each cluster

        Returns:
            List[dict]: Descriptive statistics for each cluster
        """
        cluster_descriptions = []
        for i in range(self.n_clusters):
            cluster_data = self.X[self.labels_ == i]
            cluster_desc = {
                "cluster_id": i,
                "num_series": len(cluster_data),
                "mean_series": np.mean(cluster_data, axis=0),
                "std_series": np.std(cluster_data, axis=0),
            }
            cluster_descriptions.append(cluster_desc)

        return cluster_descriptions

    def optimize_clusters(
        self, X: np.ndarray, min_clusters: int = 4, max_clusters: int = 20
    ) -> Dict[str, List[Union[int, float]]]:
        """
        Find optimal number of clusters using inertia and silhouette score.

        Args:
            X (np.ndarray): Preprocessed time series data
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try

        Returns:
            Dict with optimization results
        """
        # Reset optimization results
        self.optimization_results = {
            "n_clusters": list(range(min_clusters, max_clusters + 1)),
            "inertia": [],
            "silhouette_scores": [],
        }

        # Preprocess data if not already done
        if X.ndim == 2:
            X = self.preprocess_data(X)

        # Try different numbers of clusters
        for n_clusters in self.optimization_results["n_clusters"]:
            # Create distance parameters for DTW with LB_Keogh
            distance_params = {
                "window": self.warping_window,
                "use_lb": True,
                "lb_method": "keogh",
            }

            # Create and fit clusterer
            clusterer = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric=self.clusterer.metric,
                n_init=self.clusterer.n_init,
                max_iter=self.clusterer.max_iter,
                random_state=RANDOM_SEED,
                distance_params=distance_params,
                averaging_method="dba",
            )

            # Fit and get labels
            labels = clusterer.fit_predict(X)

            # Store inertia
            self.optimization_results["inertia"].append(clusterer.inertia_)

            # Calculate silhouette score (only if more than one cluster)
            if n_clusters > 1:
                # Flatten the time series for silhouette score calculation
                X_flat = X.reshape(X.shape[0], -1)
                sil_score = silhouette_score(X_flat, labels)
                self.optimization_results["silhouette_scores"].append(sil_score)
            else:
                self.optimization_results["silhouette_scores"].append(0)

        return self.optimization_results

    def plot_cluster_optimization(self):
        """
        Plot elbow curve and silhouette scores for cluster optimization.
        """
        if not self.optimization_results["n_clusters"]:
            raise ValueError("Run optimize_clusters() first")

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow curve (Inertia)
        ax1.plot(
            self.optimization_results["n_clusters"],
            self.optimization_results["inertia"],
            marker="o",
        )
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method")

        # Silhouette scores
        ax2.plot(
            self.optimization_results["n_clusters"],
            self.optimization_results["silhouette_scores"],
            marker="o",
            color="red",
        )
        ax2.set_xlabel("Number of Clusters")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")

        plt.tight_layout()
        plt.show()

    def recommend_clusters(self, method: str = "elbow") -> int:
        """
        Recommend optimal number of clusters.

        Args:
            method (str): Method to use for recommendation ('elbow' or 'silhouette')

        Returns:
            int: Recommended number of clusters
        """
        if not self.optimization_results["n_clusters"]:
            raise ValueError("Run optimize_clusters() first")

        if method == "elbow":
            # Find the "elbow" point where the rate of decrease in inertia slows down
            inertia = self.optimization_results["inertia"]
            # Calculate the rate of change
            inertia_diff = np.diff(inertia)
            elbow_index = np.argmax(np.abs(inertia_diff)) + 1
            return self.optimization_results["n_clusters"][elbow_index]

        elif method == "silhouette":
            # Find the number of clusters with the highest silhouette score
            sil_scores = self.optimization_results["silhouette_scores"]
            max_sil_index = np.argmax(sil_scores)
            return self.optimization_results["n_clusters"][max_sil_index]

        else:
            raise ValueError("Method must be 'elbow' or 'silhouette'")
