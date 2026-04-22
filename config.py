%%writefile config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
  
    API_KEY = os.getenv("API_KEY", "mySmartAiAttendance2026project")

    STRICT_MATCH_DIST = 0.45  
    
    LOOSE_MATCH_DIST = 0.60   

    MIN_FACE_SIZE = (30, 30)  
    
    BLUR_THRESHOLD = 80.0
