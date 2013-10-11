#! /usr/bin/python
# decision-tree.py

class TreeBuilderHeuristic(object):
    """
    interface for decision tree building heuristics
    """

    def create_node(self, features, data, parent):
        """
        return a node, which can be a feature or a label
        features -- set of features to choose from
        data -- dataset to classify
        parent -- parent node of node to be create
        """
        # if there are no unused features or no data left, then this node
        # must be a leaf (label) node
        if len(features) == 0 or len(data) or \
            self.make_leaf(features, data, parent):

            return DecisionTreeNode(parent.left_label, True)

        else:
            feature, left_label, right_label = \
                self._choose_feature(features, data, parent)

           return DecisionTreeNode(feature, False, left_label, right_label)

    def _choose_feature(self, features, data, parent):
        """
        choose the feature for the node
        this should be implemented by subclasses
        """
        pass

    def _make_leaf(features, data, parent):
        """
        should this node be a leaf?
        this should be implement by subclasses
        """
        return False


class ID3(TreeBuilderHeuristic):
    """
    heuristic for building a decision tree:
    choose the feature that maximizes information gain as
    the next parent node
    """
    # the node should be made a leaf if its parent's entropy is lower than this
    ENTROPY_THRESHOLD = 0.90

    def __init__(self):
        pass

    @classmethod
    def entropy(cls, data):
        """
        calculate the entropy of a dataset by calculating
        the frequency of the labels
        """
        pass

    def choose_feature(self, features, data, parent):
        pass

    def make_leaf(self, features, data, parent):
        return parent.entropy <= ENTROPY_THRESHOLD


class DecisionTreeNode(object):
    """
    node of a decision tree
    for now, nodes are BOOLEAN ONLY: there are only true/false edges
    this makes the tree a binary tree

    note that the children of the node either be other nodes or,
    if the node is a leaf, a classification label
    """
    def __init__(self, name="", leaf=False, left_label="", right_label="",
        left_child=None, right_child=None):

       self.name = name
       # a leaf node means this is a label
       self.leaf = leaf
       self.left = left_child
       self.right = right_child


class DecisionTree(object):
    """
    Decision tree classifier
    """

    def __init__(self, training_data, heuristic=ID3()):
        self._root = None
        # the heuristic is used to choose which features to use
        self._heuristic = heuristic
        # build the tree
        self.train(training_data)

    def train(self, data):
        """
        build the tree using training data, which consists of a 2-tuple
        of a list of features and a label
        """
        # first, create a list of features
        # this assumes that each data entry has the same features
        features = set()
        data_entry = data[0]

        for feature in  data_entry[0].keys():
            features.add(feature)

        # now build the tree!
        self._root = self._build_tree(features, data)

    def _build_tree(self, features, data, parent=None):
        """
        starting from the root, use the heuristic to recursive pick
        a feature as a node to build the tree
        """
        # choose feature returns the node created,
        # the remaining features unused, and the split of the data
        node, unused_features, left_data, right_data = \
            self._heuristic.create_node(features, data, parent)

        # if the node is not a leaf, then add subtrees under it
        if not node.leaf:
            node.left = self._build_tree(unused_features, left_data, parent)
            node.right = self._build_tree(unused_features, right_data, parent)

        return node


