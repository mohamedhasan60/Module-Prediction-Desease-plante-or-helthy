import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask_cors import CORS

# تحميل قاموس العلاجات من ملف JSON
with open('treatments.json', 'r', encoding='utf-8') as f:
    treatments = json.load(f)

@app.route('/', methods=['GET'])
def index():
    return '''
        <h2>🌿 Plant Disease Prediction API</h2>
        <p>Send a <b>POST</b> request to <code>/predict_health</code> with an image in JPEG format.</p>
        <p>Example using <code>curl</code>:</p>
        <pre>
curl -X POST -H "Content-Type: image/jpeg" --data-binary "@your_image.jpg" https://web-production-5ff01.up.railway.app/predict_health
        </pre>
    '''


app = Flask(__name__)
CORS(app)

model = load_model('plant_disease_model.keras')

# مجلد لحفظ الصور
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict_health', methods=['POST'])
def predict_health():
    if request.content_type != 'image/jpeg':
        return jsonify({"error": "Unsupported content type"}), 400

    try:
        # استلام الصورة
        img_bytes = request.get_data()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # حفظ الصورة باسم توقيت
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, img)

        # تجهيز الصورة للموديل
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        # التنبؤ
        prediction = model.predict(img_resized)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))

        class_names = list(treatments.keys())
        predicted_class = class_names[predicted_idx]

        # العلاج المناسب
        treatment = treatments.get(predicted_class, "No treatment available")

        # رابط الصورة
        image_url = f"http://{request.host}/images/{filename}"

        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence, 3),
            "treatment": treatment,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# عرض الصور المخزنة
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
