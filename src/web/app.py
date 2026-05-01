from http.client import HTTPException

from fastapi import FastAPI
from pydantic import BaseModel

from web.predict_router import predice_router
from web.service import predict_title

app = FastAPI()



@app.get("/")
def root():
    return {"message": "欢迎使用产品标题分类服务", "docs": "/docs"}

app.include_router(predice_router)

def run_app():
    import uvicorn
    uvicorn.run("web.app:app", port=8000)