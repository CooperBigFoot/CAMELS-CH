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

from src.data_models.datamodule import HydroDataModule


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


def load_country_data(country: str, config: ClusteringConfig) -> pd.DataFrame:
    """
    Load and preprocess data for a country using HydroDataModule.
    Returns cleaned daily time series DataFrame.
    """
    print(f"Loading data for {country}...")

    # Configure Caravanify
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
    ts_data = caravan.get_time_series()
    ts_colunms = ["streamflow", "date", "gauge_id"]
    ts_data = ts_data[ts_colunms]

    # Configure HydroDataModule for preprocessing
    data_module = HydroDataModule(
        time_series_df=ts_data,
        group_identifier="gauge_id",
        min_train_years=5,
        val_years=0,
        test_years=0,
        max_missing_pct=10,
        features=["streamflow"],
        target="streamflow",
        domain_id=country,
    )

    # Process data
    data_module.prepare_data()
    data_module.setup("fit")

    print(f"  Cleaned data for {country}")

    # Return cleaned daily data
    return data_module.train_dataset.df_sorted


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
    all_cleaned_daily = []
    for country in config.countries:
        try:
            cleaned_daily = load_country_data(country, config)
            all_cleaned_daily.append(cleaned_daily)
        except Exception as e:
            print(f"Error processing {country}: {e}")

    # Combine cleaned daily data
    combined_daily = pd.concat(all_cleaned_daily, ignore_index=True)

    # Generate weekly mean annual cycles for clustering
    ts_data_standardized, basin_ids = prepare_timeseries_data(
        df=combined_daily,
        basin_id_col="gauge_id",
        date_col="date",
        flow_col="streamflow",
        standardize=True,
    )

    # Proceed with clustering
    print(f"Prepared standardized data with shape: {ts_data_standardized.shape}")

    # Initialize clusterer
    clusterer = TimeSeriesClusterer(
        max_iter=config.max_iter,
        n_jobs=config.n_jobs,
        warping_window=config.warping_window,
    )

    # # Optimize clusters
    # print(
    #     f"Optimizing number of clusters from {config.min_clusters} to {config.max_clusters}..."
    # )
    # clusterer.optimize_clusters(
    #     ts_data_standardized, config.min_clusters, config.max_clusters
    # )

    # # Plot and save optimization results
    # plt.figure(figsize=(15, 7))
    # clusterer.plot_cluster_optimization(save_path=elbow_plot_path)
    # plt.close()
    # print(f"Saved optimization plot to {elbow_plot_path}")

    # Get recommended number of clusters
    # optimal_clusters = clusterer.recommend_clusters(method=config.optimization_method)
    # print(f"Recommended number of clusters: {optimal_clusters}")

    # Fit with recommended clusters
    print(f"Fitting clusterer with {13} clusters...")
    clusterer = TimeSeriesClusterer(
        n_clusters=13,
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
        attributes_base_dir="/workspace/CARAVANIFY",
        timeseries_base_dir="/workspace/CARAVANIFY",
        output_dir="./clustering_results",
        min_clusters=4,
        max_clusters=20,
        max_iter=75,
        n_jobs=-1,
        warping_window=0.2,
        optimization_method="elbow",
    )

    main(config)
