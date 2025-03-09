import numpy as np

class HillCipher:
    def __init__(self):
        self.key_matrix = None

    def create_key_matrix(self, key):
        # Convert key to numbers (A=0, B=1, etc.)
        key = [ord(c) - ord('A') for c in key.upper()]
        # Create 2x2 matrix
        self.key_matrix = np.array(key).reshape(2, 2)
        return self.key_matrix

    def matrix_mod_inverse(self, matrix, modulus):
        det = int(np.round(np.linalg.det(matrix))) % modulus
        det_inverse = pow(det, -1, modulus)
        adjugate = np.round(np.linalg.inv(matrix) * np.linalg.det(matrix)).astype(int)
        return (adjugate * det_inverse % modulus)

    def encrypt(self, plaintext, key):
        self.create_key_matrix(key)
        plaintext = plaintext.upper()
        # Remove non-alphabetic characters
        plaintext = ''.join(c for c in plaintext if c.isalpha())
        
        # Pad with X if odd length
        if len(plaintext) % 2 != 0:
            plaintext += 'X'
            
        # Convert text to numbers
        text_nums = [ord(c) - ord('A') for c in plaintext]
        
        # Process pairs of letters
        ciphertext = ''
        for i in range(0, len(text_nums), 2):
            pair = np.array(text_nums[i:i+2])
            encrypted_pair = np.dot(self.key_matrix, pair) % 26
            ciphertext += ''.join(chr(n + ord('A')) for n in encrypted_pair)
            
        return ciphertext

    def decrypt(self, ciphertext, key):
        self.create_key_matrix(key)
        # Calculate inverse key matrix
        inverse_key = self.matrix_mod_inverse(self.key_matrix, 26)
        
        # Convert text to numbers
        text_nums = [ord(c) - ord('A') for c in ciphertext]
        
        # Process pairs of letters
        plaintext = ''
        for i in range(0, len(text_nums), 2):
            pair = np.array(text_nums[i:i+2])
            decrypted_pair = np.dot(inverse_key, pair) % 26
            plaintext += ''.join(chr(int(n) + ord('A')) for n in decrypted_pair)
            
        return plaintext

    def crack(self, ciphertext):
        """Placeholder for genetic algorithm cracking"""
        return "HILL CRACKING IN PROGRESS" 