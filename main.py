%%writefile main.py
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from typing import List
import cv2
import numpy as np
import uvicorn
from core_engine import AIEngine
from logic_controller import AttendanceLogic
from config import Config
import contextlib

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == Config.API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")

ai_system = None
logic_system = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_system, logic_system
    ai_system = AIEngine()
    logic_system = AttendanceLogic(ai_system)
    yield

app = FastAPI(title="Smart Attendance API", lifespan=lifespan)

@app.post("/register", dependencies=[Depends(get_api_key)])
async def register(images: List[UploadFile] = File(...), student_id: str = Form(...)):
   
    processed_images = []
    
   
    for file in images:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            processed_images.append(img)
            
    if not processed_images:
        raise HTTPException(status_code=400, detail="No valid image files received.")

  
    result = ai_system.register_student(processed_images, student_id)
    
 
    if result["status"] == "failed":
        
        raise HTTPException(status_code=400, detail=result["reason"])
        
    return result

@app.post("/mark_attendance", dependencies=[Depends(get_api_key)])
async def mark_attendance(images: List[UploadFile] = File(...)):
    
    processed_images = []
    for file in images:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            processed_images.append(img)
    
    if not processed_images:
        return {"status": "failed", "reason": "No valid classroom images."}

    return {"report": logic_system.process_session(processed_images)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
