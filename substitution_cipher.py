import random
import string

class SubstitutionCipher:
    def __init__(self):
        self.alphabet = string.ascii_uppercase

    def generate_key(self):
        """Generate a random substitution key"""
        key_list = list(self.alphabet)
        random.shuffle(key_list)
        return ''.join(key_list)

    def encrypt(self, plaintext, key):
        if len(key) != 26:
            raise ValueError("Key must be 26 unique letters")
            
        # Create translation table
        trans = str.maketrans(self.alphabet, key.upper())
        return plaintext.upper().translate(trans)

    def decrypt(self, ciphertext, key):
        if len(key) != 26:
            raise ValueError("Key must be 26 unique letters")
            
        # Create reverse translation table
        trans = str.maketrans(key.upper(), self.alphabet)
        return ciphertext.upper().translate(trans)

    def crack(self, ciphertext):
        """Placeholder for genetic algorithm cracking"""
        return "SUBSTITUTION CRACKING IN PROGRESS" 