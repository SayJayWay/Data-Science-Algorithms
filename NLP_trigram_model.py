# A trigram model to generate random sentences
from collections import defaultdict
from bs4 import BeautifulSoup
import requests
import re
import random

""" Example: Use the same example as the bigram model so we can compare the
difference"""

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

# trigram model
trigrams = zip(document, document[1:], document[2:])
trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in trigrams:

    if prev == ".":                 # if the previous "word" was a period
        starts.append(current)      # then this is a start word

    trigram_transitions[(prev,current)].append(next)

def generate_using_trigrams():
    current = random.choice(starts)     # choose a random starting word
    prev = "."                          # and precede it with a '.'
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev,current)]
        next_word = random.choice(next_word_candidates)

        prev, current = current, next_word
        result.append(current)

        if current == ".":
            return " ".join(result)

print(generate_using_trigrams())
