from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model("breast_cancer_classifier.h5")
classes = ["sick", "normal", "unknown"]

# Configure upload folder
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = classes[np.argmax(pred)]
    confidence = float(np.max(pred)) * 100
    return pred_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pred_class, confidence = predict_image(filepath)
            return render_template("index.html", 
                                 prediction=pred_class, 
                                 confidence=confidence, 
                                 img_path=url_for('static', filename=filename))
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)