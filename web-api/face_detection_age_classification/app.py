import flask
import torch
import tensorflow as tf
import io
import os
import json
import numpy
import torchvision.transforms as transforms 
import ast

from PIL import Image
from keras.models import load_model
from keras_preprocessing.image import array_to_img

def get_image_from_file(file, max_size=1024):
    input_image = Image.open(io.BytesIO(file.read()))
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


app = flask.Flask(__name__)
w_h = 200
yolo_model = torch.hub.load('yolov5', 'custom', path='model/best.pt', source='local')
age_model = load_model('model/age_model.h5')


@app.post("/validate")
async def validate_user():
    
    if 'file' not in flask.request.files:
        return str(400)
    
    file = flask.request.files['file']
    
    if file.filename == '':
        return str(400)
        
    input_image = get_image_from_file(file)
    results = yolo_model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    if(len(detect_res) > 1):
        return str(1)
    if(len(detect_res) < 1 or detect_res[0].get('confidence') < 0.5):
        return str(0)

    file_name = 'temp/image.jpg'
    (width, height) = input_image.size
    left = int((width - w_h)/2)
    right = left + w_h
    new_img = input_image.crop((left, 0, right, height))
    new_img = new_img.resize((w_h, w_h))
    new_img = new_img.convert('RGB')
    new_img.save(file_name)

    image_string = tf.io.read_file(file_name)
    img = tf.image.decode_jpeg(image_string, channels=1)
    data = tf.reshape(img, [1, w_h, w_h])
    
    prediction = age_model.predict(data)
    predicted_age_range_index = numpy.array(prediction).argmax()
    
    os.remove(file_name)
    
    if(predicted_age_range_index < 3):
        return str(3)
    return str(4)

if __name__ == '__main__':
    app.run()