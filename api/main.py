from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from tensorflow import keras

model_path = os.path.abspath(r"C:\Users\A\OneDrive\Desktop\Plant_diseases_detection-main\models\model.h5")
MODEL = keras.models.load_model(model_path, compile=False)




CLASS_NAMES = ["Bacterial_spot", "healthy", "Early_blight", "Late_blight", "Septoria_leaf_spot", "Spider_mites_Two_spotted_spider_mite", 
"Target_Spot", "Tomato_YellowLeaf__Curl_Virus", "Tomato_mosaic_virus"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)