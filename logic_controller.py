%%writefile logic_controller.py
import face_recognition
import numpy as np
from config import Config

class AttendanceLogic:
    def __init__(self, ai_system):
        self.ai_system = ai_system
        print(" LOGIC CONTROLLER: Ready ")

    def process_session(self, class_images):
        
     
        session_results = {}
        
     
        if not self.ai_system.known_encodings:
            return [{"id": "System Error", "status": "FAILED", "reason": "No registered students in the database."}]

 
        for img_array in class_images:
            
            detected_faces = self.ai_system.detect_and_encode_classroom(img_array)
            
            for face_data in detected_faces:
                unknown_encoding = face_data["encoding"]
                
              
                distances = face_recognition.face_distance(self.ai_system.known_encodings, unknown_encoding)
                
                if len(distances) == 0:
                    continue
                    
                
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]
                matched_id = self.ai_system.known_ids[best_match_index]
                
                
                confidence = max(0.0, (1.0 - min_distance) * 100)
                
                
                if min_distance <= Config.STRICT_MATCH_DIST:
                    status = "PRESENT"
                elif min_distance <= Config.LOOSE_MATCH_DIST:
                    status = "FLAGGED"  
                else:
                    status = "UNKNOWN" 
                    matched_id = "Unknown"
            
                if matched_id != "Unknown":
                
                    if matched_id not in session_results:
                        session_results[matched_id] = {"status": status, "confidence": confidence}
                    else:
                        current_status = session_results[matched_id]["status"]
                        
                        if current_status == "FLAGGED" and status == "PRESENT":
                            session_results[matched_id] = {"status": status, "confidence": confidence}
                     
                        elif confidence > session_results[matched_id]["confidence"]:
                            session_results[matched_id]["confidence"] = confidence

      
        final_report = []
        for student_id, data in session_results.items():
            final_report.append({
                "id": student_id,
                "status": data["status"],
                "confidence": round(data["confidence"], 2)
            })
            
        return final_report
