from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
import os
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Load model and class names
model = tf.keras.models.load_model("saved_model_format", compile=False)
model.compile()
with open("class_names.json") as f:
    class_names = json.load(f)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print("❌ Image preprocessing error:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file uploaded"
            return render_template("index.html", error=error)

        file = request.files["image"]
        if file.filename == "":
            error = "No selected file"
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Please upload PNG, JPG, JPEG, or WEBP."
            return render_template("index.html", error=error)

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            with open(save_path, "rb") as img_file:
                img = preprocess_image(img_file)

            if img is None:
                error = "Failed to process image"
                return render_template("index.html", error=error)

            pred = model.predict(img)
            confidence = float(np.max(pred)) * 100
            class_idx = np.argmax(pred)
            
            if confidence >= 80:
                prediction = class_names[class_idx]
            else:
                prediction = "Disease not recognized (low confidence)"

            image_path = url_for('static', filename=f'uploads/{filename}')
            return render_template("index.html",
                                   prediction=prediction,
                                   confidence=f"{confidence:.2f}%",
                                   image=image_path,
                                   error=error)

        except Exception as e:
            print("❌ Prediction error:", e)
            error = "An error occurred during prediction"
            return render_template("index.html", error=error)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))