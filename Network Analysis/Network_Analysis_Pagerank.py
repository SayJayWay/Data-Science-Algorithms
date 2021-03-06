# Network Analysis for directed graphs and pagerank

""" This example is similar to that of websites that have the power to endorse
a fellow user.  In order to avoid people from creating phone accounts and
endorsing a single account, it will weigh based on how endorsed that individual
is.  This will be similar to the PageRank algorithm that Google uses to rank
websites."""


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


endorsements = [(0,1), (1,0), (0,2), (2,0), (1,2),
                (2,1), (1,3), (2,3), (3,4), (5,4),
                (5,6), (7,5), (6,8), (8,7), (8,9)]

for user in users:
    user["endorses"] = []
    user["endorsed_by"] = []

for source_id, target_id in endorsements:
    users[source_id]["endorses"].append(users[target_id])
    users[target_id]["endorsed_by"].append(users[source_id])

endorsements_by_id = [(user["id"], len(user["endorsed_by"]))
                      for user in users]

sorted(endorsements_by_id,
       key=lambda num_endorsements: num_endorsements, reverse=True)

def page_rank(users, damping=0.85, num_iters=100):

    # initially distribute PageRank evenly
    num_users = len(users)
    pr = { user["id"] : 1 / num_users for user in users }

    # this is the small fraction of PageRank
    # that each node gets each iteration
    base_pr = (1 - damping) / num_users

    for __ in range(num_iters):
        next_pr = { user["id"] : base_pr for user in users }
        for user in users:
            # distribute PageRank to outgoing links
            links_pr = pr[user["id"]] * damping
            for endorsee in user["endorses"]:
                next_pr[endorsee["id"]] += links_pr / len(user["endorses"])
        pr = next_pr   
    return pr

print(page_rank(users))
