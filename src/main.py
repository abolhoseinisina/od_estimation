import os
import math
import momepy
import visualizer
import od_estimator
import networkx as nx
import pandas, geopandas
from config import config

def create_directories():
    print('create_directories')
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])

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
    
    od_result = od_estimator.estimate_od_from_graph(graph, flows, od_nodes, od_names)
    print(f"Estimated OD Matrix for {config['roads_shapefile_flow_column']}:")
    print(od_result)

    od_result.to_csv(config['output_folder'] + '/od_matrix.csv')
    print(f'\nResults are saved in {config['output_folder'] + '/od_matrix.csv'}')
    
    visualizer.draw_graph(graph, f'{config['output_folder']}/network.jpg')
    visualizer.draw_od_chord_diagram(od_result, f'{config['output_folder']}/{config['roads_shapefile_flow_column']}_od_diagram.png')