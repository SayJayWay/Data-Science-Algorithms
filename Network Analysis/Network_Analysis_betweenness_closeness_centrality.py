# Simple Network Analysis using betweenness and closeness centrality
from collections import deque

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

# We can measure the metric of betweenness centrality which will identify
# individuals who frequently are on the shortest paths between pairs of other
# people
def shortest_paths_from(from_user):

    # a dictionary from "user_id" to *all* shortest paths to that user
    shortest_paths_to = { from_user["id"] : [[]] }

    # a queue of (previous user, next user) that we need to check.
    # starts out with all pairs (from_user, friend_of_from_user)

    frontier = deque((from_user, friend)
                     for friend in from_user["friends"])

    # keep going until we empty the queue
    while frontier:

        prev_user, user = frontier.popleft()    # remove the user who's
        user_id = user["id"]                    # first in the queue

        # because of the way we're adding to the queue,
        # necessarily we already know some shortest paths to prev_user
        paths_to_prev_user = shortest_paths_to[prev_user["id"]]
        new_paths_to_user = [path + [user_id] for path in paths_to_prev_user]

        # it's possible we already know a shortest path
        old_paths_to_user = shortest_paths_to.get(user_id, [])

        # what's the shrotest pat to here that we've seen so far?
        if old_paths_to_user:
            min_path_length = len(old_paths_to_user[0])
        else:
            min_path_length = float('inf')

        # only keep paths that aren't too long and are actually new
        new_paths_to_user = [path
                             for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_paths_to_user]

        shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user

        # add never-seen neighbors to the frontier
        frontier.extend((user, friend)
                        for friend in user["friends"]
                        if friend["id"] not in shortest_paths_to)

    return shortest_paths_to

for user in users:
    user["shortest_paths"] = shortest_paths_from(user)

for user in users:
    user["betweenness_centrality"] = 0.0

for source in users:
    source_id = source["id"]
    for target_id, paths in source["shortest_paths"].items():
        if source_id < target_id:       # don't double count
            num_paths = len(paths)      # how many shortest paths?
            contrib = 1 / num_paths
            for path in paths:
                for id in path:
                    if id not in [source_id, target_id]:
                        users[id]["betweenness_centrality"] += contrib

# we can also measure the closeness centrality in which we compare the sum of
# the shortest paths to each other user and determins the "farness" of each user
def farness(user):
    """the sum of the lengths of the shortest paths to each other user"""
    return sum(len(paths[0])
               for path in user["shortest_paths"].values())

for user in users:
    user["closeness centrality"] = 1/farness(user)

