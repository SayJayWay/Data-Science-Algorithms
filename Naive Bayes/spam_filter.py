import re
import math

# tokenizes messages into distinct words
def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)

# counts words in labeled training set of messages
# return dict where keys = words, values = [spam_count, non_spam_count]
def count_words(training_set):
    """training set consists of pairs (message,is_spam)"""
    counts = defaultdict(lambda:[0,0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

# turns counts into estimated probabilities using smoothing
# returns list of triplets: (word, probability word in spam message,
# probability word in nonspam message)
def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """ turn word_counts into list of triplets w,p(w|spam), p(w|~spam)"""
    return [(w,
             (spam+k)/(2*k + total_spams),
             (non_spam+k)/(2*k + total_non_spams))
            for w,(spam,non_spam) in counts.items()]

# Use word probabilities (Naive Bayes assumptions) to assign probabilities
# to messages
def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    # iterate through each word in our vocabulary
    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # if *word* appears in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # if *word* doesn't appear in the message
        # add the log probability of _not_ seeing it
        # which is log(1-probability of seeing it)
        else:
            log_prob_if_spam += math.log(1.0-prob_if_spam)
            log_prob_if_not_spam += math.log(1.0-prob_if_not_spam)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_not_spam = math.exp(log_prob_if_not_spam)
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)

# We can put this into a Naive Bayes Classifier:
class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # count spam and non-spam messages
        num_spam = len([is_spam
                        for message, is_spam in training_set
                        if is_spam])
        num_non_spams = len(training_set) - num_spams

        #run traning data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)
    def classify(self, message):
        return spam_probability(self.word_probs, message)

