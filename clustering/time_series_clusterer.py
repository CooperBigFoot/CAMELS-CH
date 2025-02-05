import numpy as np
from sktime.clustering.k_means import TimeSeriesKMeans
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42


class TimeSeriesClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.clusterer = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            n_init=5,
            max_iter=75,
            random_state=RANDOM_SEED,
        )
        self.id_to_index = {}
        self.labels_ = None

    def fit(self, X, series_ids):
        # Store the input data
        self.X = X

        # Map IDs to indices
        self.id_to_index = {id_: idx for idx, id_ in enumerate(series_ids)}

        # Fit the clusterer
        self.clusterer.fit(X)
        self.labels_ = self.clusterer.labels_
        return self

    def get_label_from_id(self, series_id):
        if series_id not in self.id_to_index:
            raise ValueError(f"Series ID {series_id} not found")
        idx = self.id_to_index[series_id]
        return self.labels_[idx]

    def plot_clusters(self):
        # Define distinct colors for centroids
        centroid_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

        # Create a subplot for each cluster
        for i in range(self.n_clusters):
            plt.figure(figsize=(10, 6))

            # Plot all series in cluster in gray
            cluster_series = self.X[self.labels_ == i]
            for series in cluster_series:
                plt.plot(series[0], color="gray", alpha=0.2)

            # Plot centroid in color
            plt.plot(
                self.clusterer.cluster_centers_[i, 0],
                color=centroid_colors[i % len(centroid_colors)],
                linewidth=2,
                label="Cluster Centroid",
            )

            plt.title(f"Cluster {i}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            sns.despine()

        plt.show()
