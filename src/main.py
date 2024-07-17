import os
import math
import momepy
import numpy as np
import networkx as nx
import pandas, geopandas
from config import config
from matplotlib import pyplot as plt
from scipy.optimize import minimize, least_squares

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

def graph_to_link_path_incidence(G: nx.DiGraph, od_list: list):
    print('graph_to_link_path_incidence')
    paths = [nx.shortest_path(G, o, d, 'mm_len') for o in od_list for d in od_list if o != d]
    edges = list(G.edges)
    
    num_paths = len(paths)
    num_edges = len(edges)
    
    A = np.zeros((num_edges, num_paths))
    for path_index, path in enumerate(paths):
        for edge_index, edge in enumerate(edges):
            if any(((edge[0], edge[1]) == (path[k], path[k+1])) for k in range(len(path)-1)):
                A[edge_index, path_index] = 1

    return A

def max_entropy_od_estimation(link_path_matrix, flow, num_zones, lambda_entropy: int = 1):
    print('max_entropy_od_estimation')
    num_paths = link_path_matrix.shape[1]

    link_path_matrix_reduced = link_path_matrix[link_path_matrix.sum(axis=1) > 0]
    flow_reduced = flow[link_path_matrix.sum(axis=1) > 0]

    def ls_objective(x):
        entropy_term = lambda_entropy * np.sum(x * np.log(x))
        least_squares_term = 0.5 * np.sum((link_path_matrix_reduced @ x - flow_reduced) ** 2)
        return entropy_term + least_squares_term
    
    x0 = np.ones(num_paths) / num_paths
    # result = least_squares(ls_objective, x0, bounds=(0, np.inf))
    
    bounds = [(0, np.inf)] * num_paths
    result = minimize(ls_objective, x0, bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 1000})

    od_matrix = np.zeros((num_zones, num_zones))
    index = 0
    for i in range(num_zones):
        for j in range(num_zones):
            if i != j:
                od_matrix[i, j] = result.x[index]
                index += 1

    return od_matrix

def grid_search_lambda(link_path_matrix, flow, lambda_values):
    print('grid_search_lambda')
    num_paths = link_path_matrix.shape[1]

    link_path_matrix_reduced = link_path_matrix[link_path_matrix.sum(axis=1) > 0]
    flow_reduced = flow[link_path_matrix.sum(axis=1) > 0]

    best_lambda = None
    best_score = float('inf')

    for lambda_entropy in lambda_values:
        def objective(x):
            entropy_term = lambda_entropy * np.sum(x * np.log(x))
            least_squares_term = 0.5 * np.sum((link_path_matrix_reduced @ x - flow_reduced) ** 2)
            return entropy_term + least_squares_term

        x0 = np.ones(num_paths) / num_paths
        bounds = [(0, np.inf)] * num_paths
        result = minimize(objective, x0, bounds=bounds, method='SLSQP', options={'maxiter': 1000})

        if result.success:
            score = result.fun
            if score < best_score:
                best_score = score
                best_lambda = lambda_entropy

    return best_lambda, best_score

def get_roads_graph(road_shapefile_path: str) -> nx.DiGraph:
    print('get_roads_graph')
    roads: geopandas.GeoDataFrame = geopandas.read_file(road_shapefile_path)
    roads.to_crs(config['projected_crs'], inplace=True)
    graph = momepy.gdf_to_nx(roads)
    graph = graph.to_directed()

    attributes = pandas.DataFrame(columns=['AADT', 'AADTT', 'TTPG'])
    for index, edge in enumerate(graph.edges(data=True)):
        attributes.loc[index] = [edge[2]['AADT'], edge[2]['AADTT'], edge[2]['TTPG']]
    
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
    
    return od_ids, ods['PCNAME'].values

if __name__ == "__main__":
    create_directories()
    graph, attributes = get_roads_graph(config['roads_shapefile'])
    draw_graph(graph)
    od_nodes, od_names = get_ods_graph_nodes(graph)
    flows = attributes['AADTT'].values

    link_path_matrix = graph_to_link_path_incidence(graph, od_nodes)
    
    lambda_values = [0.01, 0.1, 1.0, 10.0]
    best_lambda, best_score = grid_search_lambda(link_path_matrix, flows, lambda_values)
    print('best_lambda', best_lambda, 'best_score', best_score)

    od_matrix = max_entropy_od_estimation(link_path_matrix, flows, len(od_nodes), best_lambda)

    od_result = pandas.DataFrame(od_matrix, columns=od_names, index=od_names)
    print("Estimated OD Matrix:")
    print(od_result)

    od_result.to_csv(config['output_folder'] + '/od_matrix.csv')