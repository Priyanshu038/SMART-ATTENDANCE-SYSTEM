%%writefile security.py
import os
from cryptography.fernet import Fernet
import numpy as np

class SecurityVault:
    def __init__(self):
        print(" SECURITY VAULT: Initializing ")
       
        env_key = os.getenv("ENCRYPTION_KEY")
        
        if not env_key:
            self.key = Fernet.generate_key()
            print("WARNING: No ENCRYPTION_KEY found in .env. Generated a temporary session key.")
            print(f"Temporary Key (Save this to .env if saving to DB): {self.key.decode()}")
        else:
            self.key = env_key.encode()

        self.cipher_suite = Fernet(self.key)

    def encrypt_encoding(self, encoding_array):
    
        encoding_bytes = encoding_array.tobytes()
        
      
        encrypted_data = self.cipher_suite.encrypt(encoding_bytes)
        return encrypted_data

    def decrypt_encoding(self, encrypted_data):
       
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_data)
        
        encoding_array = np.frombuffer(decrypted_bytes, dtype=np.float64)
        return encoding_array
