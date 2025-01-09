import httpx
from . import Appfconf
from fastapi import FastAPI, Request,status
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import sys
import json


class FrzMsg(BaseModel):
    user: str
    token:str


class GWData(BaseModel):
      h: List[float] 
      t: List[float]    

class EndMsg(BaseModel):
      msg: str
          


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.requests_client = httpx.AsyncClient()
    yield
    await app.requests_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/stats")
async def get_ds_stats(request: Request):
    requests_client = request.app.requests_client
    response = await requests_client.get(Appfconf.env_burl+Appfconf.env_stat)
    return response.json()

@app.get("/dumpbuf")
async def get_ds_buffs(request: Request):
    requests_client = request.app.requests_client
    response = await requests_client.get(Appfconf.env_burl+Appfconf.env_dumpbuf)
    return response.json()




@app.get("/desc")
async def get_ds_desc(request: Request):
    requests_client = request.app.requests_client
    response = await requests_client.get(Appfconf.env_burl+Appfconf.env_desc)
    buf= response.json()

    desc=buf['dstatus']

    output={"resp":desc}

    return output

@app.get("/dspage")
async def redirect_ds_statpg():
   return RedirectResponse(Appfconf.env_burl,status_code=status.HTTP_303_SEE_OTHER)

@app.post("/train")
async def trainsign(request: Request,msg:FrzMsg):
    requests_client = request.app.requests_client
    buf={'user':msg.user,'token':msg.token}
    buf=json.dumps(buf)
    response = await requests_client.post(Appfconf.env_burl+Appfconf.env_freeze,
                                          data=buf)
    
    
    
    
    if(response.status_code==httpx.codes.CREATED):
     buf=response.json()
     n_file=buf['n_itm']
     n_byte=buf['buff_size']
     
     
     output={"resp":'FROZEN',"n_f":n_file,"bt_wrt":n_byte}

     
     
     
    else:
     
     output={"resp":'WAITING'}

     
    
    return output


@app.post("/streamdata")
async def streamdat(request: Request,stream:GWData):
    requests_client = request.app.requests_client
    buf={'h':stream.h,'t':stream.t}
    buf=json.dumps(buf)
    response = await requests_client.post(Appfconf.env_burl+Appfconf.env_send,data=buf)
    
    if(response.status_code==httpx.codes.CREATED):
     output={"resp":'CREATED',"body":response.text}
    else:
     
     output={"resp":'FROZEN',"body":response.text}
    


    return output


@app.post("/cleanbuf")
async def clean(request: Request,inp:EndMsg):
    requests_client = request.app.requests_client
    buf={'msg':inp.msg}
    buf=json.dumps(buf)
    response = await requests_client.post(Appfconf.env_burl+Appfconf.env_clean,data=buf)
    
    if(response.status_code==httpx.codes.CREATED):
     output={"resp":'CLEANED',"body":response.text}
    else:
     
     output={"resp":'FS ERR',"body":response.text}
    


    return output
