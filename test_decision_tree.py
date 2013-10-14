#! /usr/bin/python
"""
test_decision_tree.py
test the decision tree classifier
"""

from decision_tree import DecisionTree
from random import shuffle

def gender_features(name):
    """
    extract features from a name
    """
    return {
        "last_letter": name[-1],
        "first_letter": name[0]
    }

def main():
    """
    main function
    """
    with open("names/female.txt") as textfile:
        females = textfile.readlines()
        females = [name.strip() for name in females]

    with open("names/male.txt") as textfile:
        males = textfile.readlines()
        males = [name.strip() for name in males]

    female_features = [gender_features(name) for name in females]
    male_features = [gender_features(name) for name in males]
    shuffle(female_features)
    shuffle(male_features)

    train_set = [(feature, "female") for feature in female_features[:4500]] + \
        [(feature, "male") for feature in male_features[:2500]]
    test_set = [(feature, "female") for feature in female_features[4500:]] + \
        [(feature, "male") for feature in male_features[2500:]]

    tree = DecisionTree(train_set)
    print tree.evaluate(test_set)

if __name__ == "__main__":
    main()