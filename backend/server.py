import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, Response, redirect, render_template, request, url_for
from PIL import Image

import model

app = Flask(__name__)

video = cv2.VideoCapture("video.mp4")  # Open the video file.


def gen_frames():
    while True:
        success, frame = video.read()  # Read the video frame by frame.
        if not success:
            break
        else:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            shape = (original_frame.shape[1], original_frame.shape[0])
            print(shape)
            image = Image.fromarray(original_frame)
            image = np.array(image)
            prediction_image = model.predict(image, shape)
            print(prediction_image)
            prediction_image = cv2.resize(prediction_image, shape)
            combined_image = np.concatenate((image, prediction_image), axis=1)

            ret, buffer = cv2.imencode(".jpg", combined_image)
            processed_frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + processed_frame + b"\r\n"
            )


# @app.route("/")
# def index():
#     return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global video  # Use the global 'video' variable.
    video_file = request.files["video"]
    video_file.save("video.mp4")
    video = cv2.VideoCapture("video.mp4")  # Re-open the video file.
    return redirect("http://localhost:3000/video")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
