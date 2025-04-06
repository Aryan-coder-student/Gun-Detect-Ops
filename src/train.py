import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def download_dataset(config):
    rf = Roboflow(api_key=config["data"]["roboflow"]["api_key"])
    project = rf.workspace(config["data"]["roboflow"]["workspace"]).project(
        config["data"]["roboflow"]["project"]
    )
    version = project.version(config["data"]["roboflow"]["version"])
    dataset = version.download(config["data"]["roboflow"]["format"])
    return dataset.location

def train_model(data_path, config):
    model = YOLO(config["training"]["model"]["base_model"])
    model.train(
        data=f"{data_path}/data.yaml",
        epochs=config["training"]["hyperparameters"]["epochs"],
        imgsz=config["training"]["hyperparameters"]["img_size"],
        batch=config["training"]["hyperparameters"]["batch_size"],
        name=config["training"]["model"]["output_name"]
    )
    return model

def save_model(model, config):
    output_path = config["training"]["model"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    model.save(f"{output_path}/{config['training']['model']['output_name']}.pt")

if __name__ == "__main__":
    
    config = load_config()
    dataset_path = download_dataset(config)
    trained_model = train_model(dataset_path, config)
    save_model(trained_model, config)