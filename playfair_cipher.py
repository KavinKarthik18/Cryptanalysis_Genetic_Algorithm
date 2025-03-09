class PlayfairCipher:
    def __init__(self):
        self.key_matrix = None

    def generate_key_matrix(self, key):
        # Remove J from key and replace with I
        key = key.upper().replace('J', 'I')
        # Remove duplicates while maintaining order
        key = ''.join(dict.fromkeys(key))
        # Add remaining alphabet
        alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'  # Note: No J
        key += ''.join(c for c in alphabet if c not in key)
        
        # Create 5x5 matrix
        self.key_matrix = [list(key[i:i+5]) for i in range(0, 25, 5)]
        return self.key_matrix

    def find_position(self, char):
        char = 'I' if char == 'J' else char
        for i in range(5):
            for j in range(5):
                if self.key_matrix[i][j] == char:
                    return i, j
        return None

    def encrypt(self, plaintext, key):
        self.generate_key_matrix(key)
        plaintext = plaintext.upper().replace('J', 'I')
        # Remove non-alphabetic characters
        plaintext = ''.join(c for c in plaintext if c.isalpha())
        
        # Add X between double letters and at the end if odd length
        processed_text = ''
        i = 0
        while i < len(plaintext):
            if i == len(plaintext) - 1:
                processed_text += plaintext[i] + 'X'
                break
            if plaintext[i] == plaintext[i + 1]:
                processed_text += plaintext[i] + 'X'
                i += 1
            else:
                processed_text += plaintext[i:i+2]
                i += 2
        if len(processed_text) % 2 != 0:
            processed_text += 'X'

        ciphertext = ''
        for i in range(0, len(processed_text), 2):
            p1, p2 = self.find_position(processed_text[i]), self.find_position(processed_text[i+1])
            if p1[0] == p2[0]:  # Same row
                ciphertext += self.key_matrix[p1[0]][(p1[1]+1)%5] + self.key_matrix[p2[0]][(p2[1]+1)%5]
            elif p1[1] == p2[1]:  # Same column
                ciphertext += self.key_matrix[(p1[0]+1)%5][p1[1]] + self.key_matrix[(p2[0]+1)%5][p2[1]]
            else:  # Rectangle
                ciphertext += self.key_matrix[p1[0]][p2[1]] + self.key_matrix[p2[0]][p1[1]]

        return ciphertext

    def decrypt(self, ciphertext, key):
        self.generate_key_matrix(key)
        ciphertext = ciphertext.upper()
        plaintext = ''
        
        for i in range(0, len(ciphertext), 2):
            c1, c2 = self.find_position(ciphertext[i]), self.find_position(ciphertext[i+1])
            if c1[0] == c2[0]:  # Same row
                plaintext += self.key_matrix[c1[0]][(c1[1]-1)%5] + self.key_matrix[c2[0]][(c2[1]-1)%5]
            elif c1[1] == c2[1]:  # Same column
                plaintext += self.key_matrix[(c1[0]-1)%5][c1[1]] + self.key_matrix[(c2[0]-1)%5][c2[1]]
            else:  # Rectangle
                plaintext += self.key_matrix[c1[0]][c2[1]] + self.key_matrix[c2[0]][c1[1]]

        return plaintext.rstrip('X')

    def crack(self, ciphertext):
        """Placeholder for genetic algorithm cracking"""
        return "PLAYFAIR CRACKING IN PROGRESS" 