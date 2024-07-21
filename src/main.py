import os
import math
import momepy
import od_estimator
import networkx as nx
import pandas, geopandas
from config import config
from matplotlib import pyplot as plt

def create_directories():
    print('create_directories')
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])

def draw_graph(G):
    print('draw_graph')
    positions = {}
    for node in G.nodes():
        positions[node] = [node[0], node[1]]

    plt.figure(figsize=(10, 20))
    nx.draw(graph, positions, node_size=5, node_color="b", edge_color="grey")
    plt.savefig(f'{config['output_folder']}/network.jpg', bbox_inches='tight', dpi=100)

def get_roads_graph(road_shapefile_path: str) -> nx.DiGraph:
    print('get_roads_graph')
    roads: geopandas.GeoDataFrame = geopandas.read_file(road_shapefile_path)
    roads.to_crs(config['projected_crs'], inplace=True)
    graph = momepy.gdf_to_nx(roads)
    graph = graph.to_directed()

    attributes = pandas.DataFrame(columns=[config['roads_shapefile_flow_column']])
    for index, edge in enumerate(graph.edges(data=True)):
        attributes.loc[index] = [edge[2][config['roads_shapefile_flow_column']]]
    
    return graph, attributes

def get_ods_geometry():
    ods: geopandas.GeoDataFrame = geopandas.read_file(config['ods_shapefile'])
    return ods.to_crs(config['projected_crs'])

def get_ods_graph_nodes(graph):
    print('get_ods_graph_nodes')
    ods = get_ods_geometry()
    od_coordinates = [[point.x, point.y] for point in ods['geometry']]

    nodes = graph.nodes()
    positions = {node: (node[0], node[1]) for node in nodes}
    od_ids = []
    for od_coordinate in od_coordinates:
        id = min(positions, key=lambda node: math.hypot(od_coordinate[0]-positions[node][0], od_coordinate[1]-positions[node][1]))
        od_ids.append(id)
    
    return od_ids, ods[config['ods_shapefile_zone_name_column']].values

if __name__ == "__main__":
    create_directories()
    graph, attributes = get_roads_graph(config['roads_shapefile'])
    flows = attributes[config['roads_shapefile_flow_column']].values
    od_nodes, od_names = get_ods_graph_nodes(graph)
    draw_graph(graph)
    
    od_result = od_estimator.estimate_od_from_graph(graph, flows, od_nodes, od_names)
    print(f"Estimated OD Matrix for {config['roads_shapefile_flow_column']}:")
    print(od_result)

    od_result.to_csv(config['output_folder'] + '/od_matrix.csv')
    print(f'\nResults are saved in {config['output_folder'] + '/od_matrix.csv'}')