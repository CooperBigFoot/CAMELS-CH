import ee

PROJECT = "masterthesis-450515"
PATH_TO_BOUNDING_BOX_GEOMETRY = f"projects/{PROJECT}/assets/CAMELS-CH_bounding_box"
DEM_PRODUCT = "USGS/SRTMGL1_003"
PATH_TO_DEM_DERIVED_ATTRIBUTES = f"projects/{PROJECT}/assets/DEM_based"

try:
    ee.Initialize(project=PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT)


def get_dem_based_attributes():
    # Load the bounding box geometry
    bounding_box = ee.FeatureCollection(PATH_TO_BOUNDING_BOX_GEOMETRY).geometry()

    # Load and clip the DEM
    dem = ee.Image(DEM_PRODUCT).clip(bounding_box)

    # Compute the area of the basin using the geometry's area
    area_km2 = ee.Number(bounding_box.area()).divide(1e6)

    # Compute the mean elevation
    h_mean = dem.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=bounding_box, scale=30, maxPixels=1e9
    ).get("elevation")

    # Compute the mean slope
    slope = (
        ee.Terrain.slope(dem)
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=bounding_box, scale=30, maxPixels=1e9
        )
        .get("slope")
    )

    # Compute the dominant aspect
    aspect = (
        ee.Terrain.aspect(dem)
        .reduceRegion(
            reducer=ee.Reducer.mode(), geometry=bounding_box, scale=30, maxPixels=1e9
        )
        .get("aspect")
    )

    return {
        "area_km2": area_km2,
        "h_mean": h_mean,
        "slope": slope,
        "aspect": aspect,
    }


def export_attributes():
    attributes = get_dem_based_attributes()
    # Retrieve the bounding box geometry to attach to the exported feature
    bounding_box = ee.FeatureCollection(PATH_TO_BOUNDING_BOX_GEOMETRY).geometry()
    feature = ee.Feature(bounding_box, attributes)
    task = ee.batch.Export.table.toAsset(
        collection=ee.FeatureCollection([feature]),
        description="ExportDEMAttributes",
        assetId=PATH_TO_DEM_DERIVED_ATTRIBUTES,
    )
    task.start()
    print("Export task started. Check the Earth Engine Tasks tab for progress.")


if __name__ == "__main__":
    export_attributes()
