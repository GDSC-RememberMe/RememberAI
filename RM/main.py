from fastapi import FastAPI, HTTPException
from keyword_extractor import *
from pydantic import BaseModel

app = FastAPI()

class InputText(BaseModel):
    contents: str


@app.post("/keyword/")
async def keywords(text: InputText):
    input_txt = text.dict()
    
    result = keyword(doc=input_txt['contents'])

    data = {"keyword" : result}

    return data

# @app.exception_handler(Exception)
# async def exception_handler(request, exc):
#     print(repr(exc)) # 예외 객체 출력
#     return {"detail": "Internal server error"} # 클라이언트에 반환할 메시지