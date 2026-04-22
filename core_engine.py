%%writefile core_engine.py
import cv2
import numpy as np
import face_recognition
from retinaface import RetinaFace
from config import Config
import tensorflow as tf
import gc


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError: pass

class AIEngine:
    def __init__(self):
        self.known_encodings = []
        self.known_ids = []
        print("SYSTEM READY: Strict 3-Angle Validation Mode ")

    def safe_resize(self, img, max_dim=1280):
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(img, (0, 0), fx=scale, fy=scale), scale
        return img, 1.0

    def calculate_pose(self, landmarks):
       
        le = landmarks['left_eye']   # Viewer's Left
        re = landmarks['right_eye']  # Viewer's Right
        n  = landmarks['nose']

       
        dist_nose_to_left_eye = abs(n[0] - le[0])
        dist_nose_to_right_eye = abs(re[0] - n[0])

       
        if dist_nose_to_right_eye == 0: dist_nose_to_right_eye = 0.01

       
        ratio = dist_nose_to_left_eye / dist_nose_to_right_eye

       
        if 0.65 <= ratio <= 1.55:
            return "FRONT"
        elif ratio < 0.65:
            return "LEFT_PROFILE"
        else:
            return "RIGHT_PROFILE"

    def register_student(self, images_list, student_id):
       
        temp_encodings = []
        captured_angles = []
        
        print(f"Processing {len(images_list)} images for {student_id}...")

        for i, img_array in enumerate(images_list):
            working_img, _ = self.safe_resize(img_array)
            
            resp = RetinaFace.detect_faces(working_img, threshold=0.5)
            
            if not isinstance(resp, dict) or len(resp) != 1:
                return {
                    "status": "failed", 
                    "reason": f"Image #{i+1}: Could not find a single clear face."
                }

            key = list(resp.keys())[0]
            face_data = resp[key]
            
            angle = self.calculate_pose(face_data['landmarks'])
            captured_angles.append(angle)
            
       
            x1, y1, x2, y2 = face_data['facial_area']
            rgb = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb, [(y1, x2, y2, x1)])
            
            if encs:
                temp_encodings.append(encs[0])
            
            del working_img, rgb


        unique_angles = set(captured_angles)
        
        has_front = "FRONT" in unique_angles
        has_left  = "LEFT_PROFILE" in unique_angles
        has_right = "RIGHT_PROFILE" in unique_angles

       
        if len(images_list) >= 3:
            if not (has_front and has_left and has_right):
              
                missing = []
                if not has_front: missing.append("FRONT")
                if not has_left: missing.append("LEFT SIDE")
                if not has_right: missing.append("RIGHT SIDE")
                
                return {
                    "status": "failed",
                    "reason": f"Strict Mode Failed! You uploaded: {captured_angles}. You are MISSING: {', '.join(missing)}. Please retake."
                }
        
        
        elif len(images_list) == 2:
            if not (has_front and (has_left or has_right)):
                return {
                    "status": "failed",
                    "reason": "For 2 photos, you need 1 Front and 1 Side view."
                }

        self.known_encodings.extend(temp_encodings)
        self.known_ids.extend([student_id] * len(temp_encodings))
        
        return {
            "status": "success",
            "id": student_id,
            "angles_saved": captured_angles,
            "message": "Perfect Angle Diversity! Registration Complete."
        }

    def detect_and_encode_classroom(self, class_img_array):
        safe_img, scale = self.safe_resize(class_img_array)
        resp = RetinaFace.detect_faces(safe_img, threshold=0.4)
        
        data = []
        if isinstance(resp, dict):
            for key in resp:
                identity = resp[key]
                x1, y1, x2, y2 = identity['facial_area']
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                
                if (x2-x1) < Config.MIN_FACE_SIZE[0]: continue
                
                rgb_original = cv2.cvtColor(class_img_array, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_original, [(y1, x2, y2, x1)])
                if encs: 
                    data.append({"loc": (y1, x2, y2, x1), "encoding": encs[0], "confidence": identity['score']})
        gc.collect()
        return data
