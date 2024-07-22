import pandas as pd
import networkx as nx
import openchord as ocd
from matplotlib import pyplot as plt

def draw_od_chord_diagram(od_matrix: pd.DataFrame, output_path: str):
    print('draw_od_chord_diagram')
    fig = ocd.Chord(od_matrix.values, od_matrix.columns)
    fig.padding = 100
    fig.save_png(output_path)

def draw_graph(graph: nx.DiGraph, output_path: str):
    print('draw_graph')
    positions = {}
    for node in graph.nodes():
        positions[node] = [node[0], node[1]]

    plt.figure(figsize=(10, 20))
    nx.draw(graph, positions, node_size=5, node_color="b", edge_color="grey")
    plt.savefig(output_path, bbox_inches='tight', dpi=100)