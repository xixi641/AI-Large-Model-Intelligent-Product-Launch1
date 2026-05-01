from http.client import HTTPException

from fastapi import APIRouter

from web.shcemas import PredictRequest, PredictResponse
from web.service import predict_title

predice_router = APIRouter()

@predice_router.post("/predict")
def predict(req:PredictRequest) -> PredictResponse:
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=401, detail="请输入商品标题")
    label = predict_title(title)
    return PredictResponse(title=title,label=label)
