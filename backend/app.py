import os
import time
import base64
import cv2
import numpy as np
import joblib
import mediapipe as mp
from math import atan2, degrees, acos
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow CORS for Vercel deployment
CORS(app, resources={r"/*": {"origins": "*"}})

# Load Models
class DummyModel:
    def predict_proba(self, features):
        return [[0.0, 1.0]]  # Mock output: 100% good posture

MODELS = {}
model_files = {
    "mlp": "models/rf_mlp_pipeline.joblib",
    "rf": "models/ensemble_pipeline.joblib",
    "voting": "models/voting.joblib",
    "stacking": "models/stacking.joblib"
}

print("--- Loading ML models ---")
for key, path in model_files.items():
    try:
        if os.path.exists(path):
            MODELS[key] = joblib.load(path)
            print(f"  [OK] Loaded {key} from {path}")
        else:
            print(f"  [SKIP] {path} not found -> using DummyModel placeholder")
            MODELS[key] = DummyModel()
    except Exception as e:
        print(f"  [FAIL] Failed to load {path}: {e} -> using DummyModel placeholder")
        MODELS[key] = DummyModel()
print(f"--- Models ready (default: mlp) ---\n")

current_model = "mlp"
bad_since = None

MODEL_PATH = "models/pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = None
try:
    if os.path.exists(MODEL_PATH):
        # Using IMAGE mode instead of VIDEO mode since we process stateless frames from API
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
        )
        landmarker = PoseLandmarker.create_from_options(options)
    else:
        print(f"Warning: MediaPipe model {MODEL_PATH} not found. Keypoint extraction disabled.")
except Exception as e:
    print(f"Failed to load MediaPipe model: {e}")

VISIBILITY_THRESHOLD = 0.3

def find_inclination(x1, y1, x2, y2):
    return degrees(atan2(abs(x2 - x1), abs(y2 - y1)))

def three_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return degrees(acos(cos))

def extract_features(kp, h, w):
    ear = np.array(kp["ear"])
    shoulder = np.array(kp["shoulder"])
    hip = np.array(kp["hip"])

    neck_inc = find_inclination(*shoulder, *ear)
    torso_inc = find_inclination(*hip, *shoulder)
    sh_dist = np.linalg.norm(shoulder - hip) + 1e-6
    es_dist = np.linalg.norm(ear - shoulder)

    return np.array([[
        neck_inc,
        torso_inc,
        es_dist / sh_dist,
        (shoulder[1] - ear[1]) / h,
        (shoulder[0] - hip[0]) / w,
        three_point_angle(ear, shoulder, hip),
        find_inclination(*hip, *shoulder),
        neck_inc / (torso_inc + 1e-6),
        abs(ear[0] - hip[0]) / w,
        (hip[1] - shoulder[1]) / h,
    ]])


@app.route("/set_model", methods=["POST"])
def set_model():
    global current_model
    data = request.json
    if data and "model" in data and data["model"] in MODELS:
        current_model = data["model"]
    return jsonify({"ok": True, "current_model": current_model})

@app.route("/process_frame", methods=["POST"])
def process_frame():
    global bad_since

    data = request.json
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode base64 image
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        if landmarker:
            results = landmarker.detect(mp_image)
        else:
            class MockResults:
                pose_landmarks = []
            results = MockResults()

        latest_status = "none"
        latest_prob = 0.0
        latest_side = "none"
        latest_label = "No Detection"
        keypoints = None

        if results.pose_landmarks:
            lms = results.pose_landmarks[0]

            left_vis = lms[7].visibility + lms[11].visibility + lms[23].visibility
            right_vis = lms[8].visibility + lms[12].visibility + lms[24].visibility

            if right_vis >= left_vis:
                kp = {
                    "ear": lms[8],
                    "shoulder": lms[12],
                    "hip": lms[24],
                    "nose": lms[0],
                }
            else:
                kp = {
                    "ear": lms[7],
                    "shoulder": lms[11],
                    "hip": lms[23],
                    "nose": lms[0],
                }

            def get(lm):
                return np.array([lm.x * w, lm.y * h, lm.visibility])

            ear = get(kp["ear"])
            shoulder = get(kp["shoulder"])
            hip = get(kp["hip"])
            nose = get(kp["nose"])

            if min([ear[2], shoulder[2], hip[2], nose[2]]) > VISIBILITY_THRESHOLD:
                features = extract_features(
                    {
                        "ear": ear[:2],
                        "shoulder": shoulder[:2],
                        "hip": hip[:2],
                    },
                    h,
                    w,
                )

                model = MODELS[current_model]
                prob = model.predict_proba(features)[0][1]

                latest_prob = float(prob)
                latest_status = "good" if prob > 0.5 else "bad"
                latest_label = "GOOD POSTURE" if prob > 0.5 else "BAD POSTURE"

                if latest_status == "bad":
                    if bad_since is None:
                        bad_since = time.time()
                else:
                    bad_since = None

                torso_center_x = (shoulder[0] + hip[0]) / 2
                dx = nose[0] - torso_center_x

                if abs(dx) < 15:
                    latest_side = "center"
                elif dx < 0:
                    latest_side = "left"
                else:
                    latest_side = "right"
            else:
                bad_since = None
                
            # Return normalized keypoints for the frontend to draw
            keypoints = {
                "ear": {"x": kp["ear"].x, "y": kp["ear"].y},
                "shoulder": {"x": kp["shoulder"].x, "y": kp["shoulder"].y},
                "hip": {"x": kp["hip"].x, "y": kp["hip"].y}
            }
        else:
            bad_since = None

        bad_duration = round(time.time() - bad_since, 1) if bad_since is not None else 0

        return jsonify({
            "status": latest_status,
            "prob": latest_prob,
            "side": latest_side,
            "label": latest_label,
            "bad_duration": bad_duration,
            "keypoints": keypoints
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)