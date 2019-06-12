# Simple Network Analysis using eigenvector centrality
import numpy as np
import random
import math

users = [
    {"id":0, "name": "Hero" },
    {"id":1, "name": "Dunn" },
    {"id":2, "name": "Sue" },
    {"id":3, "name": "Chi" },
    {"id":4, "name": "Thor" },
    {"id":5, "name": "Clive" },
    {"id":6, "name": "Hicks" },
    {"id":7, "name": "Devin" },
    {"id":8, "name": "Kate" },
    {"id":9, "name": "Klein" },
    ]

friendships = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4),
               (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

for user in users:
    user["friends"] = []

for i,j in friendships:
    users[i]["friends"].append(users[j]) # Add i as a friend of j
    users[j]["friends"].append(users[i]) # Add j as a friend of i

# need to be able to convert between matrix and vector (list of lists)
def vector_as_matrix(v):
    """ returns the vector v (represented as a list) as a n x 1 matrix"""
    return [[v_i] for v_i in v]

def vector_from_matrix(v_as_matrix):
    """ returns the n x 1 matrix as a list of values"""
    return [row[0] for row in v_as_matrix]

def matrix_operate(A,v):
    v_as_matrix = vector_as_matrix(v)
    product = np.matmul(A, v_as_matrix)
    return vector_from_matrix(product)

def find_eigenvector(A, tolerance=0.1):
    guess = [random.random() for __ in A]

    while True:
        result = matrix_operate(A, guess)
        length = math.sqrt(np.dot(result, result))
        next_guess = np.multiply(1/length, result)

        if math.sqrt(np.dot(guess, next_guess)) < tolerance:
            return next_guess,  length          # eigenvector, eigenvalue

        guess = next_guess

def entry_fn(i,j):
    return 1 if (i,j) in friendships or (j,i) in friendships else 0

n =  len(users)

def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
        whose (i,j)th entry is entry_fn(i,j)"""
    return [[entry_fn(i,j)                  # given i, create a list
             for j in range(num_cols)]      # [entry_fn(i,0),....]
            for i in range(num_rows)]       # create one list for each i
adjacency_matrix = make_matrix(n,n,entry_fn)

eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)
