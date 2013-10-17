#! /usr/bin/python
"""
id3.py
heuristic for choosing features for a decision tree
"""

from __future__ import division
from math import log


def freq_dist(items):
    """
    return the frequency distribution of a list of items
    """
    num = len(items)
    # find the proportion of the labels within the dataset
    return dict([(item, len([item2 for item2 in items if item2 == item]) / num)
        for item in set(items)])


def entropy(labels):
    """
    calculate entropy from a list of probabilities
    """
    return -sum([(p * log(p, 2)) for p in labels.values()])


def id3(features, dataset):
    """
    choose the feature that minimizes entropy within the dataset
    """
    feature_entropy_list = []

    for feature in features:
        feature_entropy = 0.0

        # iterate through each of the feature's possible values
        for value in feature.values:
            # retrieve only data entries with a specific value for this feature
            filtered_dataset = [data for data in dataset
                if data[0][feature.name] == value]

            # extract a list of labels from the filtered dataset
            feature_labels = [data[1] for data in filtered_dataset]

            # calculate the entropy of the frequency dist of the labels
            # of the filtered dataset
            ent = entropy(freq_dist(feature_labels))

            # find weighted average of the feature's entropy across its
            # possible values
            ent = ent * len(filtered_dataset) / len(dataset)
            feature_entropy += ent

        feature_entropy_list.append((feature, feature_entropy)) 

    # return the feature with the least entropy
    return min(feature_entropy_list, key=lambda tup: tup[1])[0]
