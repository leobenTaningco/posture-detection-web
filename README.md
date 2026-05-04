# Posture Detection System

A real-time posture detection system using MediaPipe Pose Landmarker and Scikit-Learn models.

## 🚀 Architecture

- **Backend**: Flask API (Python) deployed on Render.
- **Frontend**: Vanilla JS / HTML / CSS app deployed on Vercel.

## 📦 Folder Structure

- `backend/`: Python Flask API, ML models, and MediaPipe logic.
- `frontend/`: Standalone static web application.

## 🛠️ Local Setup

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The API will run on `http://localhost:5000`.

### 2. Frontend
Open `frontend/index.html` in your browser or serve it using:
```bash
cd frontend
python -m http.server 3000
```
Visit `http://localhost:3000`.
