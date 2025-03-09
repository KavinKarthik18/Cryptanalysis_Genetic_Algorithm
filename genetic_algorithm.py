"""
    Using Genetic Algorithm (GA) to crack Polyalphabetic Cipher (Vigenere)
    Author: Sadip Giri (sadipgiri@bennington.edu)
    Date: 27th Nov. 2018 
"""

import random
import string
from global_variables import english_language_relative_frequencies, reverse_dict
from cryptography_algebra import relative_frequencies_calculator
from ngram_score import NgramScore
import numpy as np
import pandas as pd
from caesar_cipher import CaesarCipher
from vigenere_cipher import VigenereCipher
from playfair_cipher import PlayfairCipher
from hill_cipher import HillCipher
from substitution_cipher import SubstitutionCipher

# Initialize cipher objects
vigenere = VigenereCipher()

# loading the fitness before calculating fitness_score
fitness = NgramScore('english_quadgrams.txt')

def generate_random_keys(number_of_keys, key_length):
    """
        return list of randomly generated keys using random_key() helper function
    """
    lst = []
    for i in range(number_of_keys):
        lst.append(random_key(key_length))
    return lst

def random_key(key_length):
    """
        using random choice to randomly generate english alphabet
        return randomly generated keyword of size key_length
    """
    key = ''
    for i in range(key_length):
        key = key + random.choice(string.ascii_letters).upper()
    return key

def crossover(parent_1, parent_2):
    """
        get random crossover point
        swap alphabets between two parent keys after the crossover point 
    """
    key_size = len(parent_1)
    crossover_point = random.randint(1, key_size-1)
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return [child_1, child_2]

def mutation(parent_1, parent_2):
    """
        get two random numbers from parent_1 and parent_2
        swap alphabet indexed at two random numbers between two parents
    """
    key_size = len(parent_1)
    first_random_point  = random.randint(0, key_size-1)
    second_random_point = random.randint(0, key_size-1)
    temp_parent1 = list(parent_1)
    temp_parent2 = list(parent_2)
    temp_parent1[first_random_point] = parent_2[second_random_point]
    temp_parent1[second_random_point] = parent_2[first_random_point]
    temp_parent2[first_random_point] = parent_1[second_random_point]
    temp_parent2[second_random_point] = parent_1[first_random_point]
    return [''.join(temp_parent1), ''.join(temp_parent2)]

def fitness_score(decrypted_text):
    return fitness.score(decrypted_text.replace(' ', '').upper())

def run_genetic_algorithm(text, cipher_type='caesar', callback=None):
    """Run genetic algorithm for the specified cipher type"""
    try:
        population_size = 100
        generations = 100
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population based on cipher type
        if cipher_type == 'caesar':
            population = [random.randint(0, 25) for _ in range(population_size)]
        elif cipher_type == 'vigenere':
            population = [''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 8))) 
                       for _ in range(population_size)]
        elif cipher_type == 'playfair':
            population = [''.join(random.sample(string.ascii_uppercase.replace('J', ''), 25))
                       for _ in range(population_size)]
        elif cipher_type == 'hill':
            population = [''.join(random.choices(string.ascii_uppercase, k=4))
                       for _ in range(population_size)]
        else:  # substitution
            population = [''.join(random.sample(string.ascii_uppercase, 26))
                       for _ in range(population_size)]

        best_fitness = 0
        best_key = None
        
        for generation in range(generations):
            # Calculate fitness for each member
            fitness_scores = [calculate_fitness(key, text, cipher_type) for key in population]
            
            # Find best solution
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_key = population[max_fitness_idx]
            
            # Send update through callback
            if callback:
                update_data = {
                    'generation': generation,
                    'populationSize': population_size,
                    'bestFitness': best_fitness,
                    'averageFitness': sum(fitness_scores) / len(fitness_scores),
                    'bestKey': best_key
                }
                callback(update_data)
            
            # Selection and crossover
            new_population = []
            while len(new_population) < population_size:
                parent1 = tournament_selection(population, fitness_scores)
                parent2 = tournament_selection(population, fitness_scores)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2, cipher_type)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1, parent2])
            
            # Mutation
            population = [mutate(member, mutation_rate, cipher_type) for member in new_population[:population_size]]
            
        return best_key
        
    except Exception as e:
        print(f"Error in genetic algorithm: {str(e)}")
        return None

def keywords_and_suitability_score(keywords, cipher_text):
    """
        get fitness scores of each keyword
        return two lists with keywords and fitness_scores respectively
        this is for successfully getting top suitable keywords from following mentioned function!
    """
    key_fitness_scores = []
    for i in keywords:
        key_fitness_scores.append(fitness_score(vigenere.decrypt(ciphertext=cipher_text, key=i).upper())) 
    return [keywords, key_fitness_scores]

def top_suitable_keywords(number_of_items, keywords_with_fitness_scores):
    """
        creating pandas dataframe with keywords and fitness_scores column
        return keywords with top fitness_scores
    """
    df = pd.DataFrame(data={'keywords': keywords_with_fitness_scores[0], 'fitness_scores': keywords_with_fitness_scores[1]})
    sorted_df = df.sort_values(by=['fitness_scores'], ascending=False)
    return list(sorted_df['keywords'])[:number_of_items]

def pair_keywords(keywords_list):
    """
        shuffle the list of keywords 
        to randomly pair the keywords
        [could use roulette wheel to pair the keywords!]
    """
    np.random.shuffle(keywords_list)
    return np.array(keywords_list).reshape(int(len(keywords_list)/2), 2)

def crossover_and_certain_percent_mutation(keywords_pairs, mutation_percent):
    """
        applying crossover to keywords_pairs
        applying mutation to certain percent of crossovered children
        this is a helper function for genetic algo function!
    """
    mutation_rate = int(len(keywords_pairs) * mutation_percent)
    for i in keywords_pairs:
        children_after_crossover = crossover(parent_1=i[0], parent_2=i[1])  
        keywords_pairs = np.concatenate((keywords_pairs, np.array([children_after_crossover])), axis=0)
    np.random.shuffle(keywords_pairs)
    for i in range(mutation_rate):
        keywords_pairs[i] = mutation(keywords_pairs[i][0], keywords_pairs[i][1])
    return keywords_pairs

class GeneticAlgorithm:
    def __init__(self, ciphertext, cipher_type='caesar', population_size=100, generations=100):
        self.ciphertext = ciphertext.upper()
        self.cipher_type = cipher_type
        self.population_size = population_size
        self.generations = generations
        self.fitness_scorer = NgramScore('english_quadgrams.txt')
        
        # Initialize cipher objects
        self.ciphers = {
            'caesar': CaesarCipher(),
            'vigenere': VigenereCipher(),
            'playfair': PlayfairCipher(),
            'hill': HillCipher(),
            'substitution': SubstitutionCipher()
        }
        
        self.current_generation = 0
        self.best_fitness = 0
        self.best_key = None
        self.population = self.initialize_population()

    def initialize_population(self):
        """Initialize population based on cipher type"""
        if self.cipher_type == 'caesar':
            return [random.randint(0, 25) for _ in range(self.population_size)]
        elif self.cipher_type == 'vigenere':
            return [''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 8))) 
                   for _ in range(self.population_size)]
        elif self.cipher_type == 'playfair':
            return [''.join(random.sample(string.ascii_uppercase.replace('J', ''), 25))
                   for _ in range(self.population_size)]
        elif self.cipher_type == 'hill':
            return [''.join(random.choices(string.ascii_uppercase, k=4))
                   for _ in range(self.population_size)]
        else:  # substitution
            return [''.join(random.sample(string.ascii_uppercase, 26))
                   for _ in range(self.population_size)]

    def calculate_fitness(self, key):
        """Calculate fitness score for a key"""
        try:
            plaintext = self.ciphers[self.cipher_type].decrypt(self.ciphertext, key)
            return self.fitness_scorer.score(plaintext)
        except:
            return float('-inf')

    def select_parents(self, population, fitness_scores):
        """Select parents using tournament selection"""
        tournament_size = 5
        parents = []
        for _ in range(2):
            tournament = random.sample(list(enumerate(population)), tournament_size)
            winner = max(tournament, key=lambda x: fitness_scores[x[0]])
            parents.append(winner[1])
        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover based on cipher type"""
        if self.cipher_type == 'caesar':
            # Average the shift values
            child = (parent1 + parent2) // 2
            return child
        else:
            # Single point crossover for string keys
            point = random.randint(1, len(parent1) - 1)
            child = parent1[:point] + parent2[point:]
            
            # For substitution cipher, ensure no duplicate letters
            if self.cipher_type == 'substitution':
                used_letters = set(child[:point])
                remaining_letters = [c for c in parent2[point:] if c not in used_letters]
                unused_letters = [c for c in string.ascii_uppercase if c not in used_letters]
                random.shuffle(unused_letters)
                child = child[:point] + ''.join(remaining_letters + unused_letters)[:26-point]
            
            return child

    def mutate(self, key):
        """Mutate key based on cipher type"""
        if self.cipher_type == 'caesar':
            return (key + random.randint(-2, 2)) % 26
        elif self.cipher_type == 'substitution':
            # Swap two random positions
            key = list(key)
            i, j = random.sample(range(len(key)), 2)
            key[i], key[j] = key[j], key[i]
            return ''.join(key)
        else:
            # Replace a random character
            key = list(key)
            pos = random.randint(0, len(key) - 1)
            key[pos] = random.choice(string.ascii_uppercase)
            return ''.join(key)

    def run(self, callback=None):
        """Run the genetic algorithm"""
        generation = 0
        population = self.population
        best_fitness = float('-inf')
        best_key = None
        
        while generation < self.generations:
            # Calculate fitness for all keys
            fitness_scores = [self.calculate_fitness(key) for key in population]
            
            # Update best solution
            max_fitness_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_key = population[max_fitness_idx]
            
            # Create new generation
            new_population = []
            elite_size = self.population_size // 10
            
            # Elitism
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), 
                                                    key=lambda pair: pair[0], reverse=True)]
            new_population.extend(sorted_population[:elite_size])
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                # Selection
                parents = self.select_parents(population, fitness_scores)
                
                # Crossover
                child = self.crossover(parents[0], parents[1])
                
                # Mutation (10% chance)
                if random.random() < 0.1:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
            generation += 1
            
            # Calculate statistics for visualization
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            if callback:
                callback({
                    'generation': generation,
                    'populationSize': self.population_size,
                    'bestFitness': best_fitness,
                    'averageFitness': avg_fitness,
                    'bestKey': str(best_key)
                })
        
        # Return best solution
        return self.ciphers[self.cipher_type].decrypt(self.ciphertext, best_key)

if __name__ == '__main__':
    ft_txt_2 = "FFTWJETYJOTEZMNHSTQLTGJXILFVMBQWNGXXWMNHSFJMMHIBKMMXUHUNQTYBTGNLYHWXRTNGYAJLFFJLNSJTHKJTYNWXRNXMGXWXRHAXIMTFFDJTUEFVJYTKJTHAHANEIMMXWXFKJLJOJKFEXNHARXYATWXHSXNLYHUEFVJMMXHANEIKJGNGYAJITIZEFMNHSTYKFGIHRKJIQTHBSZFGDHSXYANLNLHTQEJWWTSWTFWXUEFVJFJGY"
    web_txt_3 = "WQPZIMKJFRSMQXJKRBHWPJIFZWBYLJHHJJWFNXJKRNAXIKHJBXIATPLYMWXJKRJOX"
    sadip_txt_5 = "SMRLTDOIMKGLXBXGNDTHGNHMSKAFPXDDLVHWRWQDFMHBWGDLNIZESWEMLDBXGNLAIGRHUPANWPTKAPMHAZHIRJEDBJJEPCHLBHZTEOYMSLOPIZWASTPUEIWGWAFPRZIOLIZEUMPJEVMKWRDTHMCKUTLHRLHGNHQHLOSTPUEWPTUHLTSJEQQCLHHXDHUOIIAOQIIJAQLDERHXASCLVVSNBWCWTKQHASFIADEGZPFDRUGWPOIRWMHVI"
    crypto_txt_6 = "CDMSXZQWCKHZWKGDGONJMCXSFJYRAWNUGCLSTKGDGAGKFDWWHKFTICRLJPMWQEGHMCTVKPBBVYCHTAGJGOXOEICPMITVKJLHDVPTFCXVBIHACBCPIZCTCUHFGRAWVVKCBIASTVYGXGGMCGTZULAWFSVYMSLCPVGHMCRCYRXHJVAWBZFICCBBVYCEHDWCYIBCPRRGTBFFKGXDNRAXGUCEWDGSVYGHBGERJAXRTRLSHATVNATQGDCCM"
    hiiamgood_txt_9 = "HUWDQRCTHCWTUFOCBDSAWNQKRGDJPQLPOBGHYBQOZSSHKVLQFFNSDRWCTAFOCBLZBWRQSOWQAPMSMSSGLGMICDKOHXYMUUEZPSULUWVQJHCPHSMABROQHMWZEMIVQKPTLTTKFSDYMAEHKFOOZCKHYKHVRKAWNQOGHRWTICQZVSFOQTDDKBWQAPMPAVIZDAQWNMZFOQKWURQVZOFPVOAZECBHAPQSUYQOOSMLRMTRCPYMXLMISAHUB"
    cryptography_txt_12 = "CDMSXZUWEKVJWKGDGORJOCLCFJYRAWRUICZCTKGDGAKKHDKGHKFTICVLLPAGQEGHMCXVMPPLVYCHTAKJIOLYEICPMIXVMJZRDVPTFCBVDIVKCBCPIZGTEUVPGRAWVVOCDIOCTVYGXGKMEGHJULAWFSZYOSZMPVGHMCVCARLRJVAWBZJIECPLVYCEHDACAIPMPRRGTBJFMGLNNRAXGUGEYDUCVYGHBGIRLALBTRLSHAXVPAHAGDCCM"
    okletsdothisand_txt_14 = "OWZHXDRTXCWDUGWYYEEKRBXLLKAPVSWHBFVSKAQGNZSDSSWAIHALXGPHZKEMHFLGMVZWMNWXELXKDAXZQRENQBPEMMUSFBALBRFOXSOWGHHTICENDVLGXXRFXHKZCUWVOXAWUSTYMKEISBLPLMFVFLBZOQGYYIBKWCISIUEGVONLBDGFXUQFTUSZZTNDDHBVVSTEOXOSFJHDEHKANTOXJSGWWVBZQKCNZVPHKSQRHTZWPYOMPQXFW"
    okletsdothisand_txt_15 = "OWZHXDRTXCWDUGLCXLPLGQSXKASCULZNTRLWUHBVVEEGKCNTJMZHDHWCDAGLCXTWMGUSFHQFTUHGKXILACSTJZWAGXFOXYLLESKLUGVRGHYXEDWDDEHKWFBUSKNLVZLZWAPWRRDFODIOWUOEZCUHZHHRZHLGQSBZBGPYDQOELXUKWEKZWNVQHRPTHHXZTAQGNNWFKYHHEUSISIUIAJOXJSGWWVBZQKCNOZOOVTFGCFYMHLNFSWPRM"
    print(run_genetic_algorithm(text=web_txt_3))