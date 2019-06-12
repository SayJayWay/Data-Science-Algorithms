# Topic modeling
import random
from collections import Counter

""" Example: Given a list of topics, weigh them and separate them into K topics
(here 4 topics) """

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learnings", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learnings", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learnings", "Big Data", "artifical intelligence"],
    ["Hadopp", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learnings", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

def sample_from(weights):
    """ returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total*random.random()      # uniform between 0 and total
    for i,w in enumerate(weights):
        rnd -= w                    # return the smallest i such that
        if rnd <= 0: return i       # weights[0] + ... + weights[i] >= rnd



# count times each topic is assigned to each document
document_topic_counts = [Counter() for _ in documents]

# count times each word is assigned to each topic
topic_word_counts = [Counter() for _ in range(4)]

# number of words to each topic
topic_counts = [0 for _ in range(4)]

# number of words in each document
document_lengths = list(map(len,documents))

# number of distinct words
distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

# number of documents
D = len(documents)

# define our conditional probability functions
def p_topic_given_document(topic, d, alpha = 0.1):
    """ the fraction of words in document _d_
        that are assigned to _topic_ (plus smoothing)"""

    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + 4 * alpha))

def p_word_given_topic(word,topic,beta=0.1):
    """the fraction of words assigned to _topic_
        that equal _word_ (plus smoothing)"""

    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W *beta))

# create weights for updating topics
def topic_weight(d,word,k):
    """Given a document and a word in it, return the weight
        for kth topic"""
    return p_word_given_topic(word,k) * p_topic_given_document(k,d)

def choose_new_topic(d,word):
    return sample_from([topic_weight(d,word,k)
                        for k in range(4)])

document_topics = [[random.randrange(4) for word in documents]
                   for documents in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

for iter in range(1000):
    for d in range(D):
        for i, (word,topic) in enumerate(zip(documents[d],
                                             document_topics[d])):
            # remove this word / topic from the counts
            # so that is doesn't influence the weights
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # choose a new topic based on the weights
            new_topic = choose_new_topic(d,word)
            document_topics[d][i] = new_topic

            # and now add it back to the counts
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count>0: print(k, word, count)

