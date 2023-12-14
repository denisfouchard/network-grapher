import argparse

#####################
# Argument parsing
#####################

parser = argparse.ArgumentParser(description="Heat graph plotting.")
parser.add_argument("file", help="Adjacency list file path (must be a .csv).")
parser.add_argument("output", help="Output file path (must be a .png)")
parser.add_argument(
    "-s",
    "--scores",
    help="Compute the PageRank score of the nodes.",
    action="store_true",
)
parser.add_argument("-p", "--plot", help="Plot the graph.", action="store_true")

args = parser.parse_args()
config = vars(args)

adj_path = config["file"]
compute_scores = config["scores"]
plot_graph = config["plot"]
output_path = config["output"]

#####################
# Algorithm
#####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

adj_list_file = pd.read_csv(adj_path, index_col=0, header=None)
# Drop rows that have NaN index
adj_list_file = adj_list_file[adj_list_file.index.notnull()]
adj_list_file = adj_list_file[adj_list_file[1].notnull()]

index_dict = {}
for i in range(len(adj_list_file)):
    index_dict[adj_list_file.index[i]] = i

name_dict = {}
for i in range(len(adj_list_file)):
    name_dict[i] = adj_list_file.index[i]

n = len(adj_list_file)
adj_list = []
for i in range(n):
    l = adj_list_file.iloc[i].dropna().values.tolist()
    if len(l) > 0:
        adj_list.append([index_dict[i] for i in l])

adj_matrix = np.zeros((n, n))
for i in range(n):
    for j in adj_list[i]:
        adj_matrix[i, j] = 1

G = nx.from_numpy_array(adj_matrix)
G = nx.relabel_nodes(G, name_dict)

if compute_scores:
    from sknetwork.ranking import PageRank

    pr = PageRank()
    scores = pr.fit_transform(adj_matrix)
    scores = scores.ravel()
    # Draw nodes with color based on PageRank score
    nx.draw_networkx(G, with_labels=True, node_size=10, font_size=1, node_color=scores)


else:
    # Draw nodes
    nx.draw_networkx(G, with_labels=True, node_size=20, font_size=4)

plt.savefig(output_path, dpi=1000)

print(f"Plot figure saved to {os.getcwd()}/" + output_path + ".")
