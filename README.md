#  Smart AI Attendance System 

A highly accurate, privacy-focused biometric attendance system built with **FastAPI**, **RetinaFace**, and **FaceNet**. Designed for real-world classroom environments, this system handles backbenchers, poor lighting, and side profiles by enforcing a strict multi-angle registration protocol.



##  Key Features

* **Strict Pose Validation:** The engine mathematically calculates the Yaw (head turn angle) during registration, forcing users to upload a diverse set of photos (Front, Left Profile, Right Profile) for maximum accuracy.
* **RetinaFace Core:** Replaced traditional HOG models with state-of-the-art RetinaFace detection. It easily identifies faces turned up to 90 degrees and excels in crowded group photos.
* **Auto-Scaling Engine:** Automatically resizes heavy 4K back-camera photos down to safe processing dimensions while retaining high-res crops for facial encoding.
* **Smart Logic Controller:** Compares classroom faces against *all* registered angles of a student, dynamically upgrading their status to `PRESENT` if even one frame matches.
* **Enterprise Security:** Built-in **Fernet Encryption** (`security.py`) to safely encrypt biometric arrays, ensuring GDPR compliance.
* **Microservice Architecture:** Cleanly decoupled into 5 distinct modules for scalability and easy debugging.



##  System Architecture (The 5 Microservices)

| Module | File | Role & Description |
| :--- | :--- | :--- |
| **Gateway** | `main.py` | FastAPI server handling HTTP requests, file batching, and API key authentication (`x-api-key`). |
| **Brain** | `core_engine.py` | Runs RetinaFace, handles auto-resizing, measures facial geometry (Pose Yaw), and generates 128-d FaceNet embeddings. |
| **Manager** | `logic_controller.py` | The business logic. Calculates Euclidean distances, matches faces against the multi-angle database, and outputs the final report. |
| **Rulebook** | `config.py` | Central configuration for thresholds (`STRICT_MATCH`, `LOOSE_MATCH`), minimum face sizes, and environment variables. |
| **Vault** | `security.py` | Handles the encryption/decryption of biometric numpy arrays before they hit the database. |



## 🛠️ Tech Stack

* **Framework:** Python 3.10+ / FastAPI / Uvicorn
* **AI/ML:** `retina-face`, `face_recognition` (dlib), `tensorflow`, `opencv-python-headless`
* **Security:** `cryptography` (Fernet)
* **Deployment:** Google Colab Ready (via PyNgrok)

