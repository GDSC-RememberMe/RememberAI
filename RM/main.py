from fastapi import FastAPI, Query
from keyword_extractor import *
from pydantic import BaseModel
import requests

app = FastAPI()

class InputText(BaseModel):
    text: str
print("test")
@app.post("/keyword")
async def keywords(text: InputText):
    input_txt = text.dict()

    result = keyword(sentence=input_txt['text'])

    url = "내부서버 url"
    data = {"keyword" : result}
    response = requests.post(url, json=data)

    return {"result" : result}