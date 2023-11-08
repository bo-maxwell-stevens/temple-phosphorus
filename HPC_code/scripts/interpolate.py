import os
import rasterio
import numpy as np
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from multiprocessing import Pool, cpu_count

# Set up the paths
gdb_path = "../Spatial/GDB/Resiel Precision Maps.gdb"
shapefiles = [('16A', 'Plots_c16A_FeatureToPoint'), ('6-12', 'Plots_c6_12_FeatureToPoint'), 
              ('Y10', 'Plots_Y10_FeatureToPoint'), ('Y8', 'Plots_Y8_FeatureToPoint'), 
              ('SW16', 'Plots_SW16_FeatureToPoint')]
output_folder = "../Data/Interpolated_soil_all"

# Depths you're interested in
depths = ['0_2', '2_6']

# List of columns to interpolate
columns_to_interpolate = ['TOC']
# , 'TC', 'IC', 'TN', 'DI Al', 'DI As', 'DI Ca', 'DI Fe', 'DI K',
#                           'DI Mg', 'DI Mn', 'DI P', 'DI S', 'DI Zn', 'DI NO3N', 'DI SRP',
#                           'DI NH4N ', 'H3A Al', 'H3A As', 'H3A Ca', 'H3A Fe', 'H3A K', 'H3A Mg',
#                           'H3A Mn', 'H3A P', 'H3A S', 'H3A Zn', 'H3A NO3', 'H3A P color',
#                           'H3A NH4N', 'M3 Al', 'M3 As', 'M3 Ca ', 'M3 Fe', 'M3 K', 'M3 Mg',
#                           'M3 Mn', 'M3 P', 'M3 S', 'M3 Zn']

# List of columns to interpolate
columns_to_interpolate = [column.replace(' ', '_') for column in columns_to_interpolate]  # Same as before

# Create a list of all columns considering the depths
all_columns = [f"{col}_{depth}" for depth in depths for col in columns_to_interpolate]

def interpolate_shapefile(shapefile):
    shapefile_name, shapefile_layer = shapefile
    gdf = gpd.read_file(gdb_path, layer=shapefile_layer)
    
    for column in all_columns:
        if gdf[column].isnull().all():
            continue
            
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        values = gdf[column].values

        xmin, ymin, xmax, ymax = gdf.total_bounds
        x_grid, y_grid = np.meshgrid(np.linspace(xmin, xmax, num=100), np.linspace(ymin, ymax, num=100))
        
        OK = OrdinaryKriging(x, y, values, variogram_model='linear', verbose=False, enable_plotting=False)
        z, ss = OK.execute('grid', x_grid, y_grid)

        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, z.shape[1], z.shape[0])

        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{shapefile_name}_interpolated_{column}.tif")
        with rasterio.open(output_file, 'w', driver='GTiff', height=z.shape[0], width=z.shape[1],
                           count=1, dtype=z.dtype, crs='+proj=latlong', transform=transform) as dst:
            dst.write(z, 1)
        print(f"Interpolated results for {column} saved to {output_file}")

if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        p.map(interpolate_shapefile, shapefiles)