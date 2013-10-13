#! /usr/bin/python
"""
decision_tree.py
Decision tree classifier
"""

from id3 import id3

class Feature(object):
    """
    data feature (i.e., metadata)
    this is used as nodes for the decision tree
    """

    def __init__(self, name, values):
        self._name = name
        self._values = values.copy()

    @property
    def name(self):
        """
        getter for name
        """
        return self._name

    @property
    def values(self):
        """
        getter for values set
        """
        return self._values.copy()


class DecisionTreeEdge(object):
    """
    edge of a decision tree
    edges contain (1) a reference to a destination node (if any),
    (2) a reference to the source/parent node, (3) and the label associated with
    the edge
    """

    def __init__(self, source, label, dest=None):
        self.source = source
        self.label = label
        self.dest = dest


class DecisionTreeNode(object):
    """
    node of a decision tree
    for now, nodes are BOOLEAN ONLY: there are only true/false edges
    this makes the tree a binary tree

    note that the children of the node either be other nodes or,
    if the node is a leaf, a classification label
    """
    def __init__(self, feature):
        self._name = feature.name
        self._edges = {}

    @property
    def name(self):
        """
        getter for name
        """
        return self._name

    @property
    def values(self):
        """
        return a list of possible values for the feature
        """
        return self._edges.keys()

    def add_edge(self, value, label, dest=None):
        """
        create an edge
        """
        edge = DecisionTreeEdge(self, label, dest)
        self._edges[value] = edge
        return edge

    def edge(self, value):
        """
        return the edge associated with a feature value
        """
        return self._edges.get(value)


class DecisionTree(object):
    """
    Decision tree classifier
    """

    def __init__(self, dataset, heuristic=id3):
        self._root = None
        self._labels = None
        self._features = None
        # the heuristic is used to choose which features to use
        self._heuristic = heuristic

        # build the tree
        self.train(dataset)

    def _partition_dataset(self, feature, dataset):
        """
        split the dataset according to the feature value that occurs
        in each data entry
        """
        partition = {}

        # build empty lists for each possible value
        for value in feature.values:
            partition[value] = {}
            partition[value]["dataset"] = []
            partition[value]["label"] = ""

        # iterate through the dataset and categorize the entries
        for data in dataset:
            partition[data[0][feature.name]].append(data)

        # assign a label for each subset
        for value in feature.values:
            val_dataset = partition[value]["dataset"]

            # build a list of tuples containing labels and the number
            # of data entries with that label
            label_count = [
                (label, len([data for data in val_dataset if data[1] == label]))
                for label in self._labels
            ]
            # return the label with the most data entries associated with it
            label = max(label_count, lambda tup: tup[1])[0]
            partition[value]["label"] = label

        return partition

    def _build_labels(self, dataset):
        """
        build a set of labels from the dataset
        """
        # clear the previous set of labels, if there was one
        self._labels = set()
        for data in dataset:
            self._labels.add(data[1])

    def _build_features(self, dataset):
        """
        build feature descriptions from the training data
        """
        # clear the previous set of features, if there was one
        self._features = set()

        # fair assumption that there's a least one entry in the dataset
        # and that the feature set is uniform across the dataset
        # (i.e., all data entries have the same feature set)
        first = dataset[0]

        # find all features
        for feature_name in first[0].keys():
            values = set()

            # find all possible values of a feature by looking at all the 
            # feature values that occur in the dataset
            for data in dataset:
                values.add(data[feature_name])

            # now add the feature to the feature set
            feature = Feature(feature_name, values)
            self._features.add(feature)

    def train(self, dataset):
        """
        build the tree using training data, which consists of a 2-tuple
        of a list of features and a label
        """
        # build set of labels
        self._build_labels(dataset)
        # build feature descriptions
        self._build_features(dataset)

        # recursive build the decision tree, starting from the root
        self._root = self._build_tree(self._features, dataset)

    def _build_tree(self, features, dataset):
        """
        use the heuristic to recursive pick a feature as a node
        do this recursively
        """
        # if there are no unused features or if the partitioned dataset is
        # empty, then don't build another node
        if len(features) == 0 or len(dataset) == 0:
            return None

        # choose a feature using the heuristic and build a node
        feature = self._heuristic(features, dataset)
        unused_features = features.copy()
        unused_features.remove(feature)
        node = DecisionTreeNode(feature)

        # split the data by feature value
        partition = self._partition_dataset(feature, dataset)

        # if only one subset of the partition has any data entries in it,
        # then there's no more need to create subtrees
        build_subtrees = sum([1 if len(partition[value]) > 0 else 0
            for value in partition.keys()]) > 1

        # build the feature's edges
        for value in partition.keys():
            edge = node.add_edge(value, partition[value]["label"])
            if build_subtrees:
                edge.dest = self._build_tree(unused_features, 
                    partition[value]["dataset"])

        return node

    def classify(self, data):
        """
        given a set of features, return a label
        """
        
        node = self._root
        while (node.edge(data[node.name]).dest is not None):
            node = node.edge(data[node.name]).dest

        return node.edge(data[node.name]).label
