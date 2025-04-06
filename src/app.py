from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import io
import yaml
import uvicorn

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
app = FastAPI()
model = YOLO(f"{config['training']['model']['output_path']}/{config['training']['model']['output_name']}.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform prediction
    results = model.predict(source=img, conf=config["training"]["hyperparameters"]["confidence_threshold"])
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf)
            cls = int(box.cls)
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "class": model.names[cls]
            })
    
    return {"detections": detections}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["reload"]
    )