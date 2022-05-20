# coding: utf-8
#!/usr/bin/python3
#!flask/bin/python3
import os
import uvicorn

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

## Reading Scores and CEFR Level
import reading
from cefr_predictor.inference import Model
model = Model("cefr_predictor/models/xgboost.joblib")


main = FastAPI()

@main.get("/readscore/{text}")
async def readscore(text):
    text = text.replace("/","")
    result = reading.get_stat(text)
    return result 

@main.post("/predict/{texts}")
def predict(texts):
    preds, probas = model.predict_decode([texts])
    response = []
    for text, pred, proba in zip([texts], preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        response.append(row)
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)    


