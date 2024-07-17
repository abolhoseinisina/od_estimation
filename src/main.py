import os
import math
import momepy
import numpy as np
import networkx as nx
import pandas, geopandas
from config import config
from matplotlib import pyplot as plt
from scipy.optimize import minimize, least_squares

def draw_graph(G):
    print('draw_graph')
    positions = {}
    for node in G.nodes():
        positions[node] = [node[0], node[1]]

    plt.figure(figsize=(10, 20))
    nx.draw(graph, positions, node_size=5, node_color="b", edge_color="grey")
    plt.tight_layout()
    plt.savefig(f'{config['output_folder']}/network.jpg', dpi=100)

def graph_to_link_path_incidence(G: nx.DiGraph, od_list: list):
    paths = [nx.shortest_path(G, o, d) for o in od_list for d in od_list if o != d]
    edges = list(G.edges)
    
    num_paths = len(paths)
    num_edges = len(edges)
    
    A = np.zeros((num_edges, num_paths))
    for path_index, path in enumerate(paths):
        for edge_index, edge in enumerate(edges):
            if any(((edge[0], edge[1]) == (path[k], path[k+1])) for k in range(len(path)-1)):
                A[edge_index, path_index] = 1

    return A

def max_entropy_od_estimation(link_path_matrix, flow, num_zones):
    num_paths = link_path_matrix.shape[1]

    link_path_matrix_reduced = link_path_matrix[link_path_matrix.sum(axis=1) > 0]
    flow_reduced = flow[link_path_matrix.sum(axis=1) > 0]

    def ls_objective(x):
        lambda_entropy = 1
        entropy_term = lambda_entropy * np.sum(x * np.log(x))
        least_squares_term = 0.5 * np.sum((link_path_matrix_reduced @ x - flow_reduced) ** 2)
        return entropy_term + least_squares_term
    
    x0 = np.ones(num_paths) / num_paths
    result = least_squares(ls_objective, x0, bounds=(0, np.inf))
    
    bounds = [(0, np.inf)] * num_paths
    # result = minimize(ls_objective, x0, bounds=bounds, method='SLSQP', options={'disp': True})

    od_matrix = np.zeros((num_zones, num_zones))
    index = 0
    for i in range(num_zones):
        for j in range(num_zones):
            if i != j:
                od_matrix[i, j] = result.x[index]
                index += 1

    return od_matrix

if __name__ == "__main__":
    G = nx.graph_atlas(150)
    G = G.to_directed()
    flows = np.array([randint(1, 10) for i in range(len(G.edges()))])
    draw_graph(G, flows)

    od_nodes = [0, 1, 2]
    link_path_matrix = graph_to_link_path_incidence(G, od_nodes)
    od_matrix = max_entropy_od_estimation(link_path_matrix, flows, len(od_nodes))

    print("Estimated OD Matrix:")
    print(od_matrix)