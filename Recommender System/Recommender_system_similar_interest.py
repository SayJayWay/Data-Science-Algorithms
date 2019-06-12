# Recommender system based on interests that are similar to the user's interest
from collections import defaultdict
import numpy as np
import math


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

# use cosine_similarity to determine how similar two interests are
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


# transpose the matrix so rows = intersts and columns = users
interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_matrix]
                        for j,_ in enumerate(unique_interests)]


# Since matrix is transposed, we can calculate similarities between interests
interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                      for user_vector_j in interest_user_matrix]
                     for  user_vector_i in interest_user_matrix]

# Find the most similar interests to interest_id
def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0 ]
    return sorted(pairs,
                  key=lambda similarity:similarity, reverse = True)

# Create recommendation for a user by summing up the similarities of interests
# similar to user
def item_based_suggestions(user_id, include_current_interests = False):
    # add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # sort them by weight
    suggestions = sorted(suggestions.items(),
                         key = lambda similarity:similarity, reverse = True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

print(item_based_suggestions(0))
