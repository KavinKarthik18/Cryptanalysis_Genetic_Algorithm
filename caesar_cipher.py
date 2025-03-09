"""
    encrypt and decrypt Caesar Cipher
    Author: Sadip Giri (sadipgiri@bennington.edu)
    Date: 3rd Nov. 2018
"""

import numpy as np
from cryptography_algebra import convert_np_array_numbers_to_sentence, convert_sentence_to_np_array_of_numbers

class CaesarCipher:
    def __init__(self):
        pass

    def encrypt(self, plaintext, key):
        """Encrypt text using Caesar cipher"""
        result = ""
        for char in plaintext.upper():
            if char.isalpha():
                # Shift character by key positions
                shifted = chr((ord(char) - ord('A') + key) % 26 + ord('A'))
                result += shifted
            else:
                result += char
        return result

    def decrypt(self, ciphertext, key):
        """Decrypt text using Caesar cipher"""
        return self.encrypt(ciphertext, (26 - key) % 26)

    def crack(self, ciphertext):
        """Simple frequency analysis to crack Caesar cipher"""
        # This is a placeholder - actual cracking is done by genetic algorithm
        return self.decrypt(ciphertext, 3)  # Default to key=3 for demo

if __name__ == '__main__':
    encrypted_text = CaesarCipher().encrypt("hi i am sadip", 10)
    print(CaesarCipher().crack(encrypted_text))