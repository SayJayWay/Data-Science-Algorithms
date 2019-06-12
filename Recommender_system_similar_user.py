# Recommender system based on users that are similar
import numpy as np
import math
from collections import defaultdict

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
    ]

# use cosine_similarity to determine how similar two users are
def cosine_similarity(v,w):
    """ relate vector v and w where if both vectors point in:
        same direction -- returns 1
        if v = 0 when w!= 0 and vice versa -- returns 0
        opposite direction -- returns -1"""
        
    return np.dot(v,w) / math.sqrt(np.dot(v,v) * np.dot(w,w))

# return a list of all the distinct interests
unique_interests = sorted(list({interest
                                for user_interests in users_interests
                                for interest in user_interests}))

# produce an interest vector depicting 0 if user does not have interest and
# 1 if user does
def make_user_interest_vector(user_interests):
    """given a list of interests, produce a vector whose ith element is 1
        if unique_interests[i] is in the list, 0 otherwise"""
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

# create matrix of user interests
user_interest_matrix = map(make_user_interest_vector, users_interests)

# calculate pairwise similarities between all users
user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                     for  interest_vector_i in user_interest_matrix]

# This function will print the most similar users to user x and list the
# 'magnitude' of similarity from 0 to 1
def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity)                        # find other                
             for other_user_id, similarity in                   # users with
             enumerate(user_similarities[user_id])              # nonzero
             if user_id != other_user_id and similarity > 0]    # similarity

    return sorted(pairs,
                  key = lambda similarity: similarity, reverse = True)

def user_based_suggestions(user_id, include_current_interests = False):
    # sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # convert them to a sorted list
    suggestions = sorted(suggestions.items(),
                         key = lambda weight:weight, reverse = True)

    # excluse already known interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

print(user_based_suggestions(0))

