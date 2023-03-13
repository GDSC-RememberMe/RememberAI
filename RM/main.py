from fastapi import FastAPI, Query
from keyword_extractor import *
from pydantic import BaseModel

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/keyword")
async def keywords(text: InputText):
    input_txt = text.dict()

    result = keyword(sentence=input_txt['contents'])

    data = {"keyword" : result}

    return data