import os
from datetime import datetime
from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model('plant_disease_model.keras')

# المكان اللي نحفظ فيه الصور
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# قاموس العلاج
treatments = {
    "Blast": "Apply fungicides like Tricyclazole or Isoprothiolane. Use disease-resistant varieties and avoid dense planting to reduce humidity.",
    # ... باقي الأمراض
}

@app.route('/predict_health', methods=['POST'])
def predict_health():
    if request.content_type != 'image/jpeg':
        return jsonify({"error": "Unsupported content type"}), 400

    try:
        img_bytes = request.get_data()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # حفظ الصورة
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, img)

        # تجهيز الصورة
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        # التنبؤ
        prediction = model.predict(img_resized)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        class_names = list(treatments.keys())
        predicted_class = class_names[predicted_idx]
        treatment = treatments.get(predicted_class, "No treatment available")

        image_url = f"http://{request.host}/images/{filename}"

        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence, 3),
            "treatment": treatment,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# لعرض الصور المخزنة
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
