from flask import request

import math
import numpy as np
import cv2

import torch

model = torch.hub.load("ultralytics/yolov5", 'custom',
                       path="./assets/model/200 64.pt")


def home():
    return {"message": "success!"}


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
    print(output)

    # Convert the model output to a suitable format for response
    response = output.to_dict('records')

    print(request.form["width"])
    print(request.form["height"])

    # Return the response as JSON
    return response
