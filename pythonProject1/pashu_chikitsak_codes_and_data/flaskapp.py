from flask import Flask, render_template, request, send_file

import requests
from PIL import Image
from io import BytesIO

from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
model = YOLO('.\\runs\\classify\\train\\weights\\best.pt')


@app.route('/')
def index():
    return render_template('index.html')


def process_image_(image_path):
    img = Image.open(image_path)
    results = model(img)
    names_dict =  results[0].names
    probs = results[0].probs.tolist()
    print("Class Names:", names_dict)
    print("Probabilities:", probs)
    print(f"for image {image_path} \t\t : {names_dict[np.argmax(probs)]}")
    return names_dict[np.argmax(probs)]


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        try:
            img = Image.open(file)
            img_path = 'uploaded_image.png'  # You can save it with a proper name
            img.save(img_path)

            result = process_image_(img_path)
            return render_template('result.html', image=img_path, prediction=result)
        except Exception as e:
            return f'<h1>Error: {str(e)}</h1>'


@app.route('/url', methods=['POST'])
def process_image_url():
    image_link = request.form.get('image_link')
    if image_link:
        try:
            response = requests.get(image_link)
            img = Image.open(BytesIO(response.content))
            img_path = 'uploaded_image.png'  # You can save it with a proper name
            img.save(img_path)

            result = process_image_(img_path)
            return render_template('result.html', image=img_path, prediction=result)
        except Exception as e:
            return f'<h1>Error: {str(e)}</h1>'
    else:
        return '<h1>No image link provided</h1>';


@app.route('/display_image/<filename>')
def display_image(filename):
    return send_file(filename, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
