# A bigram model to generate random sentences
from collections import defaultdict
from bs4 import BeautifulSoup
import requests
import re
import random

""" Example: Use a data science paragraph to get some words to use: http://oreil.ly/1Cd6ykN)
# since will be extracting the text from a website, we want Unicode character u"\u2019" to actually
# be an apostrophe
"""
def fix_unicode(text):
    return text.replace(u"\u2019", "'")

# now we can import the text and clean it up a bit
url = "https://www.oreilly.com/ideas/what-is-data-science"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

content = soup.find("div", "article-body")      # find entry-content div
regex = r"[\w']+|[\.]"                          # matches a word or a period


document = []

for paragraph in content("p"):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)

# bigram model (creates a dictionary with pairs of consecutive words in a certain document
bigrams = zip(document, document[1:])
transitions = defaultdict(list)
for prev, current in bigrams:
    transitions[prev].append(current)

# generate some sentences
def generate_using_bigrams():
    current = "."   # this means the next word will start a sentence
    result = []

    while True:
        next_word_candidates = transitions[current]     # bigrams (current, _)
        current = random.choice(next_word_candidates)   # choose one at random
        result.append(current)                          # append it to results
        if current == ".": return " ".join(result)      # if ".", then end of sentence

print(generate_using_bigrams())

