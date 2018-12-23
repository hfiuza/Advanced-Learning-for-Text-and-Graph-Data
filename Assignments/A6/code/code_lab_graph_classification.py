"""
Graph Mining - ALTEGRAD - Dec 2018
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from grakel.kernels import ShortestPath, PyramidMatch, RandomWalk, VertexHistogram, WeisfeilerLehman
from grakel import graph_from_networkx
from grakel.datasets import fetch_dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Question 1
# Generate simple dataset

Gs = list()
y = list()

##################
# your code here #
##################
Gs.extend([nx.generators.classic.cycle_graph(length) for length in range(3, 103)])
y.extend([0] * 100)

Gs.extend([nx.generators.classic.path_graph(length) for length in range(3, 103)])
y.extend([1] * 100)

############## Question 2
# Classify the synthetic graphs using graph kernels

# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

##################
# your code here #
##################
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Transform NetworkX graphs to objects that can be processed by GraKeL
G_train = list(graph_from_networkx(G_train))
G_test = list(graph_from_networkx(G_test))


# Use the shortest path kernel to generate the two kernel matrices ("K_train" and "K_test")
# hint: the graphs do not contain node labels. Set the with_labels argument of the the shortest path kernel to False

##################
# your code here #
##################
gk = ShortestPath(with_labels=False)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

# Compute the classification accuracy
# hint: use the accuracy_score function of scikit-learn

##################
# your code here #
##################
print('Accuracy using shortest path kernel: {}'.format(accuracy_score(y_pred, y_test)))

# Use the random walk kernel and the pyramid match graph kernel to perform classification

##################
# your code here #
##################
gk = RandomWalk()
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print('Accuracy using random walk kernel: {}'.format(accuracy_score(y_pred, y_test)))

gk = PyramidMatch(with_labels=False)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print('Accuracy using pyramid match kernel: {}'.format(accuracy_score(y_pred, y_test)))


############## Question 3
# Classify the graphs of a real-world dataset using graph kernels

# Load the MUTAG dataset
# hint: use the fetch_dataset function of GraKeL

##################
# your code here #
##################
mutag = fetch_dataset('MUTAG', verbose=False)
G, y = mutag.data, mutag.target

# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

##################
# your code here #
##################
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1)

# Perform graph classification using different kernels and evaluate performance

##################
# your code here #
##################
gk = RandomWalk()
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print('Accuracy using random walk kernel: {}'.format(accuracy_score(y_pred, y_test)))

gk = PyramidMatch(with_labels=False)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print('Accuracy using pyramid match kernel: {}'.format(accuracy_score(y_pred, y_test)))

gk = WeisfeilerLehman(base_kernel=VertexHistogram)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

print('Accuracy using Weisfeiler-Lehman subtree kernel: {}'.format(accuracy_score(y_pred, y_test)))

# Results: better accuracy for Shortest Paths kernel, for both questions
