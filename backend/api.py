import io

import cv2
import model
import numpy as np
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

############# IMAGE FILE TEST - USING POSTMAN
# @app.post("/predict/")
# async def predict_api(file: UploadFile = File(...)):

#     image_data = await file.read()
#     image = Image.open(io.BytesIO(image_data))

#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     prediction = model.predict(image)
#     prediction = prediction.tolist()

#     return JSONResponse(content=prediction)


############# IMAGE BYTE TEST
@app.post("/predict/")
async def predict_api(image_data: bytes = Body(...)):

    image = Image.open(io.BytesIO(image_data))

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = model.predict(image)
    prediction = prediction.tolist()

    return JSONResponse(content=prediction)
