from __future__ import annotations

import io
from typing import Any

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from app.model import build_unet
from app.preprocess import mask_from_prediction, overlay_mask_on_image, preprocess_image

app = FastAPI(title="LUNA16 ROI API", version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_unet_lung_roi.pth"


def load_model() -> torch.nn.Module:
    model = build_unet()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    x = preprocess_image(image).to(device)

    with torch.no_grad():
        pred = model(x)
        pred_mask = mask_from_prediction(pred)

    payload: dict[str, Any] = {
        "prediction_shape": list(pred_mask.shape),
        "nonzero_pixels": int(pred_mask.sum()),
    }
    return JSONResponse(payload)


@app.post("/predict_mask")
async def predict_mask(file: UploadFile = File(...)) -> StreamingResponse:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    x = preprocess_image(image).to(device)

    with torch.no_grad():
        pred = model(x)
        pred_mask = mask_from_prediction(pred)

    out = Image.fromarray((pred_mask * 255).astype("uint8"))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict_overlay")
async def predict_overlay(file: UploadFile = File(...)) -> StreamingResponse:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    x = preprocess_image(image).to(device)

    with torch.no_grad():
        pred = model(x)
        pred_mask = mask_from_prediction(pred)

    overlay = overlay_mask_on_image(image, pred_mask)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
