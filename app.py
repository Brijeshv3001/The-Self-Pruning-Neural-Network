# =================================================================
# File: app.py
# Project: The Self-Pruning Neural Network
# Description: FastAPI production server
# =================================================================

import base64
import os
import csv
from io import BytesIO
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
from PIL import Image
from contextlib import asynccontextmanager

from src.model import SelfPruningNet
from src.layers import PrunableConv2d, PrunableLinear
import torchvision.transforms as transforms

class PredictRequest(BaseModel):
    image_base64: str
    top_k: int = 3

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

global_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_model
    model_path = os.getenv("MODEL_PATH", "experiments/model_lambda_0.0001.pt")
    if os.path.exists(model_path):
        global_model = SelfPruningNet(num_classes=10)
        global_model.load_state_dict(torch.load(model_path, map_location=device))
        global_model.eval()
        global_model.to(device)
    else:
        print(f"[WARNING] Checkpoint {model_path} not found. Server starting without injected weights.")
    yield
    global_model = None

app = FastAPI(title="The Self-Pruning Neural Network", lifespan=lifespan)

@app.get("/")
def read_root():
    """Renders the Aether HUD interactive operational context."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "SelfPruningNet", "device": str(device)}

@app.get("/model/info")
def model_info():
    if not global_model:
        raise HTTPException(status_code=503, detail="Neural parameters explicitly absent.")
    cr = global_model.get_compression_ratio()
    return {
        "parameters": global_model.count_parameters(),
        "sparsity": global_model.get_network_sparsity(),
        "compression_ratio": float(cr) if cr != float('inf') else 999.9,
        "gflops": global_model.compute_flops()
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not global_model:
        raise HTTPException(status_code=503, detail="Neural block uninitialized.")
    try:
        data = req.image_base64.split(",")[-1]
        img_bytes = base64.b64decode(data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((32, 32))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        input_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = global_model(input_t)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            
        top_probs, top_idxs = torch.topk(probs, req.top_k)
        
        predictions = []
        for p, idx in zip(top_probs, top_idxs):
            predictions.append({
                "class": cifar10_classes[idx.item()],
                "confidence": float(p.item() * 100.0)
            })
            
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/results")
def get_results():
    results_path = "experiments/results.csv"
    if not os.path.exists(results_path):
        return []
    res = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res.append(row)
    return res

@app.get("/model/gates")
def get_model_gates():
    if not global_model:
        return []
    stats = []
    for name, module in global_model.named_modules():
        if isinstance(module, (PrunableConv2d, PrunableLinear)):
            gates = torch.sigmoid(module.gate_scores / module.temperature)
            sparsity = 100.0 * (1.0 - (gates > 0.01).sum().item() / gates.numel())
            mean_gate = gates.mean().item()
            stats.append({"layer_name": name, "sparsity": sparsity, "mean_gate": mean_gate})
    return stats
