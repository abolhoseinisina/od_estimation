import pandas
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from sklearn.preprocessing import MaxAbsScaler

def graph_to_link_path_incidence(G: nx.DiGraph, od_list: list):
    print('graph_to_link_path_incidence')
    paths = [nx.shortest_path(G, o, d, 'mm_len') for o in od_list for d in od_list if o != d]
    edges = list(G.edges)
    
    num_paths = len(paths)
    num_edges = len(edges)
    
    link_path_incident_matrix = np.zeros((num_edges, num_paths))
    for path_index, path in enumerate(paths):
        for edge_index, edge in enumerate(edges):
            if any(((edge[0], edge[1]) == (path[k], path[k+1])) for k in range(len(path)-1)):
                link_path_incident_matrix[edge_index, path_index] = 1

    return link_path_incident_matrix

def max_entropy_od_estimation(link_path_matrix, flow, num_zones, lambda_entropy: int = 1):
    print('max_entropy_od_estimation')
    num_paths = link_path_matrix.shape[1]

    link_path_matrix_reduced = link_path_matrix[link_path_matrix.sum(axis=1) > 0]
    flow_reduced = flow[link_path_matrix.sum(axis=1) > 0]

    def objective_function(x):
        entropy_term = -np.sum(x * np.log(x + 1e-20))+(1e-20)
        least_squares_term = 0.5 * np.sum((link_path_matrix_reduced @ x - flow_reduced) ** 2)
        cost_value = least_squares_term / (lambda_entropy * entropy_term)
        return cost_value
    
    x0 = np.ones(num_paths) / num_paths
    bounds = [(0, 1)] * num_paths
    result = minimize(objective_function, x0, bounds=bounds, method='SLSQP', options={'disp': True, 'maxiter': 1000})
    
    rmse = np.sqrt(np.sum((link_path_matrix_reduced @ result.x - flow_reduced) ** 2)/ len(result.x))
    print('RMSE', rmse)

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
        def objective_function(x):
            entropy_term = -np.sum(x * np.log(x + 1e-20))+(1e-20)
            least_squares_term = 0.5 * np.sum((link_path_matrix_reduced @ x - flow_reduced) ** 2)
            cost_value = least_squares_term / (lambda_entropy * entropy_term)
            return cost_value

        x0 = np.ones(num_paths) / num_paths
        bounds = [(0, 1)] * num_paths
        result = minimize(objective_function, x0, bounds=bounds, method='SLSQP', options={'maxiter': 1000})

        if result.success:
            score = result.fun
            if score < best_score:
                best_score = score
                best_lambda = lambda_entropy

    return best_lambda, best_score

def estimate_od_from_graph(graph, flows, od_nodes, od_names):
    print('estimate_od_from_graph')
    scaler = MaxAbsScaler()
    flows = scaler.fit_transform(flows.reshape(-1, 1)).flatten()

    link_path_matrix = graph_to_link_path_incidence(graph, od_nodes)
    
    lambda_values = [0.1, 1.0, 10.0]
    best_lambda, _ = grid_search_lambda(link_path_matrix, flows, lambda_values)

    od_matrix = max_entropy_od_estimation(link_path_matrix, flows, len(od_nodes), best_lambda)
    od_matrix = scaler.inverse_transform(od_matrix.flatten().reshape(-1, 1)).reshape(od_matrix.shape)
    od_matrix = np.round(od_matrix, 0).astype(int)

    od_result = pandas.DataFrame(od_matrix, columns=od_names, index=od_names)
    return od_result