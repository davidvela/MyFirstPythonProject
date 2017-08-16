from collections import Counter
import numpy as np

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")


def proof_correlation():
    # Create three Counter objects to store positive, negative and total counts
    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()  
    