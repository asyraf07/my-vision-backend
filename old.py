from flask import Flask, request, jsonify
from numpy.lib import math
import torch
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# model = torch.hub.load("ultralytics/yolov5", 'custom',
#                        path="./assets/model/bestSkripsi.pt")
model = torch.hub.load("ultralytics/yolov5", 'custom',
                       path="./assets/model/bestCoco.pt")


@app.route("/", methods=["GET"])
def root():
    return jsonify({"text": "hello", "status": 200})


@app.route("/hello", methods=["POST"])
def hello():
    img = Image.open(request.files["image"].stream)
    img = img.convert('RGB')
    img.save("./assets/image/pillow/before.jpg")

    img = img.rotate(270, expand=True)
    img.save("./assets/image/pillow/rotate.jpg")

    width = math.ceil(float(request.form["width"]))
    height = math.ceil(float(request.form["height"]))

    scale = (height, width)
    img = img.resize(scale)
    img.save("./assets/image/pillow/resize.jpg")
    # image = request.files["image"].read()
    # image =

    return jsonify({"text": "hello", "status": 200})


@app.route("/testing", methods=["POST"])
def testing():
    image = request.files["image"].read()

    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

    dim = (392, 696)
    image = cv2.resize(image, dim)

    cv2.imwrite("./assets/image/saved.jpg", image)

    print("===============")
    print(request.form)
    print("===============")

    return {"status": "success"}


@app.route("/predict", methods=["POST"])
def predict():
    if (len(request.files) < 1):
        return {"message": "no files found!"}

    # Read width and height
    width = math.ceil(float(request.form["width"]))
    height = math.ceil(float(request.form["height"]))

    # Read the image from the request
    image = request.files["image"].read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

    cv2.imwrite("./assets/image/before.jpg", image)

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("./assets/image/rotated.jpg", image)

    dim = (width, height)
    # dim = (392, 685)

    image = cv2.resize(image, dim)

    cv2.imwrite("./assets/image/resized.jpg", image)

    # Run object detection on the image
    with torch.no_grad():
        output = model(image)

    output = output.pandas().xyxy[0]

    # Convert the model output to a suitable format for response
    response = output.to_dict('records')

    print(request.form["width"])
    print(request.form["height"])

    # Return the response as JSON
    return response


if __name__ == "__main__":
    # app.run()
    app.run(host="0.0.0.0", port=5000, debug=True)
