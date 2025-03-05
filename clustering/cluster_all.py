import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Ensure the parent directory is in the path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from clustering.preprocess_time_series import prepare_timeseries_data
from clustering.time_series_clusterer import TimeSeriesClusterer
from src.data_models.caravanify import Caravanify, CaravanifyConfig


@dataclass
class ClusteringConfig:
    """Configuration for the time series clustering process."""

    # Data sources
    countries: List[str]
    attributes_base_dir: str
    timeseries_base_dir: str

    # Clustering parameters
    min_clusters: int = 4
    max_clusters: int = 20
    max_iter: int = 75
    n_jobs: int = -1
    warping_window: float = 0.2

    # Output paths
    output_dir: str = "./clustering_results"
    elbow_plot_filename: str = "elbow_plot.png"
    cluster_plot_filename: str = "cluster_plot.png"
    results_csv_filename: str = "cluster_assignments.csv"

    # Optimization method
    optimization_method: str = "elbow"  # 'elbow' or 'silhouette'


def load_country_data(country: str, config: ClusteringConfig) -> tuple:
    """
    Load time series and static data for a specific country.

    Args:
        country: Country code (e.g., 'CH', 'CL', 'USA')
        config: Clustering configuration

    Returns:
        Tuple of (time_series_df, static_attributes_df)
    """
    print(f"Loading data for {country}...")

    caravan_config = CaravanifyConfig(
        attributes_dir=f"{config.attributes_base_dir}/{country}/post_processed/attributes",
        timeseries_dir=f"{config.timeseries_base_dir}/{country}/post_processed/timeseries/csv",
        gauge_id_prefix=country,
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    caravan = Caravanify(caravan_config)
    ids = caravan.get_all_gauge_ids()
    print(f"  Found {len(ids)} stations for {country}")

    caravan.load_stations(ids)
    return caravan.get_time_series(), caravan.get_static_attributes()


def main(config: ClusteringConfig):
    """
    Main function to cluster time series data from multiple countries.

    Args:
        config: Clustering configuration
    """
    # Create output directory if it doesn't exist
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create full paths for output files
    elbow_plot_path = output_dir / config.elbow_plot_filename
    cluster_plot_path = output_dir / config.cluster_plot_filename
    results_csv_path = output_dir / config.results_csv_filename

    # Load data from all countries
    all_ts_data = []
    all_static_data = []

    for country in config.countries:
        try:
            ts_data, static_data = load_country_data(country, config)
            all_ts_data.append(ts_data)
            all_static_data.append(static_data)
        except Exception as e:
            print(f"Error loading data for {country}: {e}")

    # Combine all data
    if not all_ts_data:
        print("No data loaded. Exiting.")
        return

    combined_ts_data = pd.concat(all_ts_data, ignore_index=True)
    print(
        f"Combined time series data: {len(combined_ts_data)} records from {len(set(combined_ts_data['gauge_id']))} stations"
    )

    # Prepare data for clustering
    ts_data_standardized, basin_ids = prepare_timeseries_data(
        df=combined_ts_data,
        basin_id_col="gauge_id",
        date_col="date",
        flow_col="streamflow",
    )

    print(f"Prepared standardized data with shape: {ts_data_standardized.shape}")

    # Initialize clusterer
    clusterer = TimeSeriesClusterer(
        max_iter=config.max_iter,
        n_jobs=config.n_jobs,
        warping_window=config.warping_window,
    )

    # Optimize clusters
    print(
        f"Optimizing number of clusters from {config.min_clusters} to {config.max_clusters}..."
    )
    clusterer.optimize_clusters(
        ts_data_standardized, config.min_clusters, config.max_clusters
    )

    # Plot and save optimization results
    plt.figure(figsize=(15, 7))
    clusterer.plot_cluster_optimization(save_path=elbow_plot_path)
    plt.close()
    print(f"Saved optimization plot to {elbow_plot_path}")

    # Get recommended number of clusters
    optimal_clusters = clusterer.recommend_clusters(method=config.optimization_method)
    print(f"Recommended number of clusters: {optimal_clusters}")

    # Fit with recommended clusters
    print(f"Fitting clusterer with {optimal_clusters} clusters...")
    clusterer = TimeSeriesClusterer(
        n_clusters=optimal_clusters,
        max_iter=config.max_iter,
        n_jobs=config.n_jobs,
        warping_window=config.warping_window,
    )
    clusterer.fit(ts_data_standardized, basin_ids)

    # Plot and save clusters
    plt.figure(figsize=(20, 15))
    clusterer.plot_clusters(max_series_per_cluster=10, save_path=cluster_plot_path)
    plt.close()
    print(f"Saved cluster plot to {cluster_plot_path}")

    # Create mapping of gauge_id to cluster
    id_to_cluster = {id_: clusterer.get_label_from_id(id_) for id_ in basin_ids}

    # Save to CSV
    results_df = pd.DataFrame(
        {
            "gauge_id": list(id_to_cluster.keys()),
            "cluster": list(id_to_cluster.values()),
        }
    )
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")


if __name__ == "__main__":
    # Define configuration
    config = ClusteringConfig(
        countries=["CH", "CL", "USA"],
        attributes_base_dir="/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY",
        timeseries_base_dir="/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY",
        output_dir="./clustering_results",
        min_clusters=4,
        max_clusters=20,
        max_iter=75,
        n_jobs=-1,
        warping_window=0.2,
        optimization_method="elbow",
    )

    main(config)
