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

#My Libraries
import reading
import words as wds
import wav_vec_transcribe as wavtrans
import ipa as ipaa
import diffcheck 
import WordLevelProblems as probcheck
import nominalize 
import formality

from pydantic import BaseModel
from typing import List
import aiofiles
import urllib.request

from cefr_predictor.inference import Model

model = Model("cefr_predictor/models/xgboost.joblib")


class TextList(BaseModel):
    texts: List[str] = []


main = FastAPI()

main.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

origins = ["*"]

main.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_file(filename, data):
    with open(filename, 'wb') as f:
        fullpath = os.path.join("static/audio/", filename)
        f.write(fullpath,data)
        
@main.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@main.get("/reading", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("reading.html", {"request": request})

@main.get("/suggestion", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("suggest.html", {"request": request})

@main.get("/audio", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("audio.html", {"request": request})

@main.get("/nominalize", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("nominalize.html", {"request": request})

@main.get("/formality", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("formality.html", {"request": request})

@main.get("/readscore/{text}")
async def readscore(text):
    text = text.replace("/","")
    result = reading.get_stat(text)
    return result 

@main.get("/suggest/{text}")
async def suggest(text):
    try:
        result = wds.get_words(text)
        print(result)
        formatted = result[0]
        suggs = result[1]
        topic = result[2]
        return formatted,suggs,topic
    except Exception as e:
        print(e)
        return "Error","<div class='alert alert-danger' role='alert'>No word suggestion. Continue to write a better sentence.</div>",""

@main.post("/predict/{texts}")
def predict(texts):
    preds, probas = model.predict_decode([texts])
    response = []
    for text, pred, proba in zip([texts], preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        response.append(row)
    return response

@main.get("/ipa/{audiofile}")
async def transcribe(audiofile):
    loc = "static/audio/{}".format(audiofile)
    result = wavtrans.get_wav2vec([loc])
    return result

@main.get("/recognition/{audiourl}")
async def speech(audiourl):
    loc = urllib.request.urlretrieve(audiourl)
    result = wavtrans.get_wavwords([loc])
    return result

@main.get("/speech/{audiofile}")
async def speech(audiofile):
    loc = "static/audio/{}".format(audiofile)
    result = wavtrans.get_wavwords([loc])
    return result

@main.get("/nominal/{text}")
async def nominal(text):
    text = "{} means ".format(text)
    result = nominalize.get_nom(text)
    return result

@main.get("/formal/{text}")
async def formal(text):
    results = []
    offensive_res = formality.get_offensive(text)
    formal_res = formality.get_formal(text)
    results.append(offensive_res)
    results.append(formal_res)
    return results

@main.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = os.path.join('static/audio/', file.filename) 
    async with aiofiles.open(filename, 'wb') as f:
        contents = await file.read()
        await f.write(contents)
        cmdtext = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 {}.wav".format(filename,filename)
        fpeg = os.system(cmdtext)
        loc = "{}.wav".format(filename)
        ipatext = wavtrans.get_wav2vec([loc])
        return ipatext

@main.get("/ipacheck/{user}/{word}")
async def ipacheck(user,word):
    print(user,word) 
    result = []
    words = word.split()
    if len(words) == 1: #One word = type 1
        #try:
            ipa_list = ipaa.get_ipa(word)
            standards = ipa_list[1][0]
            result = diffcheck.checkipa(user.replace("",""),standards)
            #result.append(("type1",difference,standards))
        #except:
            #result = ["type3"]
            #pass
    else: #Sentence = type 2
        #try:
            stanard_ipa = ipaa.get_sent_ipa(word)
            check = probcheck.ipa_analyse(word,user)
            result.append("type2")
            result.append(check[0])
            result.append(check[1])
            result.append(stanard_ipa)
        #except:
            #result = ["type3"]
            #pass
    #print("RESULT",result)
    return result



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5151)    


