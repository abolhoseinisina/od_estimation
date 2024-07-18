**Origin-Destination (OD) Matrix Estimation using Maximum Entropy and Least-Square Error**

To start using this script, you have start by creating a virtual environment:

`python -m venv [name]`

Then activate your virtual environment by running the following command inside the root folder:

`source bin/activate`

The clone this repository:

`git clone [repository_url]`

You have to install the requirements of this repository simply by the following command:

`pip install -r [requirements_file]`

Next, you are required to modify the `config.json` file to include your files. You need 2 shapefiles: 
1. A line shapefile for the road network.
2. A point shapefile for the origin destination nodes. All the points in the later shapefile are considered as origin and destination.

If your shapefiles are in a coordinate CRS instead of a projection CRS, calculation can go wrong. So, you have to provide a projection CRS inside the `config.json` file based on the geographical location of your road network. If your shapefile is in a projection CRS, include the same projection EPSG code inside the config file.

Also specify a directory so that the script saves the results of the script inside that fodler.

After following the above steps, simply run the following command:

`python main.py`

The estimated OD Matrix is going to be saved the output directory you provided.
