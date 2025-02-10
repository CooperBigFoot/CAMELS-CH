import ee

PROJECT = "masterthesis-450515"
PATH_TO_BOUNDING_BOX_GEOMETRY = f"projects/{PROJECT}/assets/CAMELS-CH_bounding_box"
DEM_PRODUCT = "USGS/SRTMGL1_003"
PATH_TO_BASIN_BOUNDARY = f"projects/{PROJECT}/assets/CAMELS_CH_sub_catchments_ESPG4326"

try:
    ee.Initialize(project=PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT)


def get_dem_based_attributes(basin_feature: ee.Feature) -> dict:
    """
    Computes DEM-based attributes for a given basin feature.

    Attributes computed:
    - Area (km²)
    - Mean elevation (m)
    - Mean slope (°)
    - Mode aspect (°)
    """
    geom = basin_feature.geometry()
    dem = ee.Image(DEM_PRODUCT).clip(geom)

    area_km2 = ee.Number(geom.area()).divide(1e6)
    h_mean = dem.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e9
    ).get("elevation")
    slope = (
        ee.Terrain.slope(dem)
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e9)
        .get("slope")
    )
    aspect = (
        ee.Terrain.aspect(dem)
        .reduceRegion(reducer=ee.Reducer.mode(), geometry=geom, scale=30, maxPixels=1e9)
        .get("aspect")
    )

    return {
        "area_km2": area_km2,
        "h_mean": h_mean,
        "slope": slope,
        "aspect": aspect,
    }


def extract_dem_attributes(feature: ee.Feature) -> ee.Feature:
    """
    Extracts gauge_id and DEM-based attributes from a basin feature,
    and removes the geometry since only gauge_id and attributes are needed.
    """
    attrs = get_dem_based_attributes(feature)
    # Assuming the original feature includes a 'gauge_id' property.
    gauge_id = feature.get("gauge_id")
    attrs["gauge_id"] = gauge_id
    return ee.Feature(None, attrs)


if __name__ == "__main__":
    basin_boundaries = ee.FeatureCollection(PATH_TO_BASIN_BOUNDARY)
    dem_attributes_collection = basin_boundaries.map(extract_dem_attributes)

    # Select only the desired properties
    dem_attributes_collection = dem_attributes_collection.select(
        ["gauge_id", "area_km2", "h_mean", "slope", "aspect"]
    )

    # Export as CSV using Drive since CSV export is not supported directly to assets.
    export_task = ee.batch.Export.table.toDrive(
        collection=dem_attributes_collection,
        description="Basins_DEM_Attributes_CSV",
        folder="gee_exports",
        fileFormat="CSV",
    )

    export_task.start()
    print("CSV export task started.")
