from collections import Counter
import numpy as np
import math

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner,_ = votes.most_common(1)
    return winner

def majority_vote(labels):
    """assumes labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner                       # unique winner, so return it
    else:
        return majority_vote(labels[:-1])    # try again w/o farthest neighbour

def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair(point,label)"""
    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key = lambda point:math.sqrt(np.dot((point, new_point))))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # let them vote
    return majority_vote(k_nearest_labels)

# Note: KNN runs into trouble in higher dimension, since they are vast.  Points in high dimensional tend not to be close
# to one another.  To visualize this, randomly generate pairs of points in d-dimensional "unit cube" in variety of
# dimensions and calculate the distance between them
                             
