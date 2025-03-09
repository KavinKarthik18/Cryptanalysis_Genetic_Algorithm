"""
    encrypt and decrypt Vigenere Cipher
    Author: Sadip Giri (sadipgiri@bennington.edu)
    Date: 31th Oct. 2018
"""

from cryptography_algebra import convert_sentence_to_np_array_of_numbers, convert_np_array_numbers_to_sentence, maintain_length_of_keyword_with_plaintext
# from genetic_algorithm import run_genetic_algorithm

class VigenereCipher:
    def __init__(self):
        pass

    def encrypt(self, plaintext, key):
        """Encrypt text using Vigenere cipher"""
        result = ""
        key = key.upper()
        key_length = len(key)
        key_as_int = [ord(i) - ord('A') for i in key]
        plaintext = plaintext.upper()
        
        for i, char in enumerate(plaintext):
            if char.isalpha():
                # Get the shift from the key (cycling through it)
                key_shift = key_as_int[i % key_length]
                # Shift the character and add it to the result
                shifted = chr((ord(char) - ord('A') + key_shift) % 26 + ord('A'))
                result += shifted
            else:
                result += char
        return result

    def decrypt(self, ciphertext, key):
        """Decrypt text using Vigenere cipher"""
        result = ""
        key = key.upper()
        key_length = len(key)
        key_as_int = [ord(i) - ord('A') for i in key]
        ciphertext = ciphertext.upper()
        
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                # Get the shift from the key (cycling through it)
                key_shift = key_as_int[i % key_length]
                # Reverse the shift and add it to the result
                shifted = chr((ord(char) - ord('A') - key_shift) % 26 + ord('A'))
                result += shifted
            else:
                result += char
        return result

    def crack(self, ciphertext):
        """Placeholder for genetic algorithm cracking"""
        return self.decrypt(ciphertext, "KEY")  # Default key for demo

# def crack(ciphertext, keylength, number_of_generations):
#     return run_genetic_algorithm(keylength, number_of_generations, ciphertext, mutation_rate=0.2)

if __name__ == '__main__':
    # print(encrypt("dCode Vigenere automatically", "KEY"))
    # print(decrypt(encrypt("dCode Vigenere automatically", "KEY"), "KEY"))
    # text1 = "Hi I am Sadip Giri from Bennignton College. I am working on cryptanalysis of Vigenere cipher using Genetic Algorithm which is very interesting heuristic approach."
    # keyword = "crypto"
    # keyword = "web"
    # keyword = "cryptography"
    keyword = "sadipgiriisgoodperson"
    txt = "A model of evolution also needs a child insertion method. If the population is to remain the same size, a creature must be removed to make a place for each child. There are several such methods. One is to place the children in the population at random, replacing anyone. This is called random replacement."
    print(encrypt(txt,"ft"))
