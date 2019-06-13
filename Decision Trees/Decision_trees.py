# Basic decision tree function
import math
import random
from collections import defaultdict
from functools import partial

# Entropy is a notion of "how much information"
# It is used to represent the uncertainty of data
def entropy(class_probabilities):
    """given a list of class probabilities, compute the entropy"""
    return sum(-p*math.log(p,2)
               for p in class_probabilities
               if p)                            # ignore 0 probabilities

# determine probabilities of each lebel
def class_probabilities(labels):
    total_count=len(labels)
    return [count/total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels=[label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

# Determine the resulting entropy from partitions
def partition_entropy(subsets):
    """ Find the entropy from this partition of data into subsets;
        subsets is a list of lists of labeled data"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset)*len(subset)/total_count
               for subset in subsets)

#using example given, start by writing a function that does the partitioning
def partition_by(inputs,attribute):
    """each input is a pair (attribute_dict, label)
        Returns a dict: attribute_value -> inputs"""
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]   # get the value of the specified attribute
        groups[key].append(input)   # then add this input to the correct list
    return groups

def partition_entropy_by(inputs, attribute):
    """ computes the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

#We can classify an input:
def classify(tree,input):
    """ classify the input using the given decision tree"""

    # if this is a leaf nose, return its value
    if tree in [True, False]:
        return tree

    # Otherwise this tree consists of an attribute to split on and a dictionary
    # keys are values of that attribute and whose values of are subtrees to
    # consider next
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)  # None if input is missing attribute

    if subtree_key not in subtree_dict: # if no subtree for key
        subtree_key = None              # we'll use the None subtree

    subtree = subtree_dict[subtree_key] # choose the appropriate subtree
    return classify(subtree,input)      # and use it to classify the input

# Build the tree representation from our training data
def build_tree_id3(inputs, split_candidates=None):
    # if this is our first pass, all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs-num_trues

    if num_trues == 0: return False     # no Trues? return "false" leaf
    if num_falses == 0: return True     # no Falses? return "true" leaf

    if not split_candidates:            # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf

    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
                         key = partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    nnum_candidates = [a for a in split_candidates
                       if a != best_attribute]

    # recursively build the subtrees
    subtrees = {attribute_value: built_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.items()}

    subtrees[None] = num_trues > num_falses     # default case

    return (best_attribute,subtrees)

""" Since decision trees have a tendecncy to overfit (as they easily fit to
their training data, we can use "random forests" to build multiple decision
trees and let them vote on how to classify inputs"""

def forest_classify(trees,input):
    votes = [classify(tree,input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

""" Since the above mentioned tree building process is deterministic, we can
bootstrap our data to achieve different trees"""
def bootstrap_sample(data):
    """ Randomly samples len(data) elements w/ replacement"""
    return [random.choice(data) for _ in data]

""" We can introduce a second source of randomness ("ensemble learning") by
changing how we choose the "best-attribute" to split on.  Instead of looking
at all the remaining attributes, first choose a random subset of them, then
split on whicever of those is the best"""
# if there is already enough split candidates, look at all of them
if len(split_candidates)<=self.num_split_candidates:
    sampled_split_candidates = split_candidates
# otherwise pick a random sample
else:
    sampled_split_candidates = random.sample(split_candidates, self.num_split_candidates)

# now choose best attribute only from those candidates
best_attribute = min(sampled_split_candidates, key=partial(partition_entropy_by, inputs))

partitions = partition_by(inputs, best_attribute)
    

