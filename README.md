**Origin-Destination (OD) Matrix Estimation using Maximum Entropy and Least-Square Error**

To start using this script, you should start by creating a virtual environment:

`python -m venv [name]`

Then activate your virtual environment by running the following command inside the root folder:

`source bin/activate`

Next, clone this repository:

`git clone https://github.com/abolhoseinisina/od_estimation.git`

Before running the script, make sure to install the requirements of this repository simply by the following command:

`pip install -r [requirements_file]`

Next, you are required to modify the `config.json` file to include your settings. You need to include paths to 2 shapefiles: 
1. A linear shapefile for your transportation network.
2. A point shapefile for your origin destination nodes. All the points in this shapefile are considered as origin and destination.

If your shapefiles are in a coordinate CRS (e.g. EPSG:4326) instead of a projection CRS (e.g. EPSG:26914), calculations can go wrong. So, you have to provide a projection CRS inside the `config.json` file based on the geographical location of your road network. If your shapefiles are already in a projection CRS, include the same projection EPSG code inside the config file: `projected_crs`
The script reprojects your shapefiles to the projection CRS you provided in the config file.

The transportation network shapefile must have a column for the flow values. You have to include the column name in the config file: `roads_shapefile_flow_column`

The origin-destinations shapefile must have a column for the name of zones. You have to include the column name in the config file: `ods_shapefile_zone_name_column`

Also, specify a directory so that the script saves the results of the script inside that folder: `output_folder`

After following the above steps, simply run the following command:

`python main.py`

The estimated OD Matrix is going to be saved the output directory you provided.
