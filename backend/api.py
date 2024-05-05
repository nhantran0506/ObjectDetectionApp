from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import cv2
import model

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
