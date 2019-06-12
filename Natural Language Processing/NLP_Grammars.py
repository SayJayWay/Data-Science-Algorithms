# Grammars
""" We can generate our own grammar rules and add a list of nouns/verbs to pick
from that will satisfy the conditions."""
import random

# Example where:
# _S = "sentence rule", _NP = "noun phrase", _VP = "verb phrase", _V = "verb"
grammar = {
    "_S" : ["_NP _VP"],
    "_NP" : ["_N",
             "_A _NP _P _A _N"],
    "_VP" : ["_V",
             "_V _NP"],
    "_N" : ["data science", "Python", "regression"],
    "_A" : ["big", "linear", "logistic"],
    "_P" : ["about", "near"],
    "_V" : ["learns", "trains", "tests", "is"]
    }

# terminal are the values in these dictionary
def is_terminal(token):
    return token[0] != "_"

def expand(grammar, tokens):
    for i, token in enumerate(tokens):

        # skip over terminals
        if is_terminal(token): continue

        # if we get here, we found a non-terminal token
        # so we need to choose a replacement at random
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

        # now call expand on the new list of tokens
        return expand(grammar, tokens)
    # if we get here we had all terminals and are done
    return tokens

def generate_sentence(grammar):
    return expand(grammar, ["_S"])

print(generate_sentence(grammar))
        
