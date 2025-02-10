import ee
import geopandas as gpd

PROJECT = "masterthesis-450515"
BASE_DIR = f"projects/{PROJECT}/assets"


try:
    ee.Initialize(project=PROJECT)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=PROJECT)

# TODO: Turn this into an actual function
# geometry = ee.Geometry.Polygon(
#     [
#         [
#             [5.860068506597615, 48.02963712004707],
#             [10.596744444353288, 48.02963712004707],
#             [10.596744444353288, 45.68298143391117],
#             [5.860068506597615, 45.68298143391117],
#             [5.860068506597615, 48.02963712004707],
#         ]
#     ],
#     proj="EPSG:4326",
# )

# # Export the geometry to project assets
# task = ee.batch.Export.table.toAsset(
#     collection=ee.FeatureCollection(geometry),
#     description="geometry",
#     assetId=f"{BASE_DIR}/CAMELS-CH_bounding_box",
# )

# task.start()