"""
Graph Mining - ALTEGRAD - Dec 2018
"""

# Import modules
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


############## Question 1
# Load the graph into an undirected NetworkX graph

##################
# your code here #
##################
graph = nx.read_edgelist(path='../datasets/CA-HepTh.txt')


############## Question 2
# Network Characteristics

##################
# your code here #
##################
print('1 - Number of nodes: {}'.format(graph.number_of_nodes()))
print('2 - Number of edges: {}'.format(graph.number_of_edges()))
print('3 - Number of connected components: {}'.format(nx.algorithms.components.number_connected_components(graph)))

############## Question 3
# Analysis of degree distribution

##################
# your code here #
##################
degree_sequence = [d for _, d in graph.degree()]
print('1 - Min degree: {}'.format(np.min(degree_sequence)))
print('2 - Max degree: {}'.format(np.max(degree_sequence)))
print('3 - Mean degree: {}'.format(np.mean(degree_sequence)))

# Plotting the histogram
y = nx.degree_histogram(graph)
plt.plot(y, 'b-', marker='o')
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.show()

# Plotting the histogram with both axes in logarithm scale. We can now observe a linearly decreasing curve
plt.loglog(y, 'b-', marker='o')
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.show()
