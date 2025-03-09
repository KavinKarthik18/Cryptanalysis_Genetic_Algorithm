"""
    fitness_score of text using n-gram statistics (probabilities)
    Used quadgram corpus from Practical Cryptography website by James Lyons which is cited below
    Date: 29th Nov. 2018
"""

import math
import os
from collections import defaultdict

class NgramScore:
    def __init__(self, ngramfile, sep=' '):
        """ Initialize with an ngram file in the format:
            ngram count
            e.g. for quadgrams:
            TION 60000
            NTHE 45000
            etc...
        """
        self.ngrams = {}
        self.L = 0  # L is length of the ngrams
        
        # Create ngrams file if it doesn't exist
        if not os.path.exists(ngramfile):
            self.create_default_ngrams(ngramfile)
        
        # Load ngram counts from file
        with open(ngramfile, 'r') as f:
            for line in f:
                key, count = line.strip().split(sep)
                self.ngrams[key] = int(count)
                self.L = len(key)
        
        # Calculate log probabilities
        self.N = sum(self.ngrams.values())
        for key in self.ngrams.keys():
            self.ngrams[key] = math.log10(float(self.ngrams[key]) / self.N)
        
        # Set floor value for ngrams not found
        self.floor = math.log10(0.01 / self.N)
    
    def score(self, text):
        """ Compute the score of text using the ngram frequencies """
        text = text.upper()
        score = 0
        ngrams = self.ngrams.__getitem__
        
        for i in range(len(text) - self.L + 1):
            if text[i:i+self.L] in self.ngrams:
                score += ngrams(text[i:i+self.L])
            else:
                score += self.floor
        
        return score
    
    def create_default_ngrams(self, filename):
        """ Create a default quadgram file if none exists """
        quadgrams = defaultdict(int)
        
        # Common English quadgrams and their approximate frequencies
        common_quadgrams = """
        TION 60000
        NTHE 45000
        THER 40000
        THAT 35000
        OFTH 30000
        FTHE 25000
        THES 20000
        WITH 15000
        INTH 12000
        ATIO 10000
        TING 9000
        SAND 8000
        IONS 7000
        MENT 6000
        THIS 5000
        HERE 4500
        FROM 4000
        OULD 3500
        IGHT 3000
        HAVE 2500
        OUGH 2000
        ANCE 1500
        WERE 1000
        THIN 900
        THEM 800
        """
        
        # Write to file
        with open(filename, 'w') as f:
            for line in common_quadgrams.strip().split('\n'):
                f.write(line.strip() + '\n')

"""
    Citation:
        Title: Quadgram Statistics as a Fitness Measure
        Author: James Lyons
        Date: 2009-2012
        Availability: http://practicalcryptography.com
"""
    




    