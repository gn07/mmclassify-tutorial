from tokenize import String
from typing import Optional
import os
from fastapi import FastAPI, File, UploadFile
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
from inference import inference_model
import uvicorn


app = FastAPI()

# upload model file here
model = models.mobilenet_v2(pretrained=True)
num_ftrs = model.classifier[1].in_features
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.classifier[1] = nn.Linear(num_ftrs, 101)
device = torch.device("cpu")
model = model.to(device)
model.load_state_dict(torch.load('food_model.pt', map_location=device))
with open('classes.txt') as f:
    classes = f.read().splitlines()

@app.get("/")
async def read_root():
    try:
        return {"Message": "Hellow World"}
    except Exception as e:
        return {"Unknown Error": str(e)}

@app.post("/predict")
async def predict(file: Optional[bytes] = File(None)):
    if file:
        results = inference_model(model,file, classes)
        print(results)
        return {
            'Message': 'Success',
            'Results': results,
            'File Size': len(file)
        }
    else:
        return {
            'Message': 'No file uploaded'
        }
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)