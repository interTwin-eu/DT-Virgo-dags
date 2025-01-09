from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from astropy.units.quantity import Quantity as q
import numpy as np
import aiohttp
import asyncio
import os

env_gwid=os.environ['GWID']
env_dtcid=os.environ['DTCID']
env_inf=float(os.environ['INF'])
env_sup=float(os.environ['SUP'])
env_srv=os.environ['SRV']

N_BUFFS=3
BUFF_SZ=300



def createBuff(pointer,arr_data,buffer):
  
  for i in range(BUFF_SZ):
     indx=pointer+i
     buffer.append(arr_data[indx])

  return indx


async def sendGW(url,gwdata):

  
  async with aiohttp.ClientSession() as session:
    async with session.post(url,json=gwdata) as response:
      
        output = await response.text()
        print(output)
      


async def parallPost(GWS):
  
  tasks = [asyncio.create_task(sendGW(env_srv,GW)) for GW in GWS]
  await asyncio.gather(*tasks)      






def main():


  f=open("lastrd.txt","r")
  last_pt=f.readline()
  f.close()
  datapt=int(last_pt)
  gps=event_gps(env_gwid)
  data=TimeSeries.fetch_open_data(env_dtcid,gps+env_inf,gps+env_sup)


  
  
  sdt=data.dt.value
  st0=0.0+sdt*(datapt+1)

  print("Size for", env_gwid,"data",": ",len(data.value)*8," Byte" )
  print ("Array length is: ",len(data.value))


  buff1=[]
  buff2=[]
  buff3=[]
  
  datapt=createBuff(datapt,data.value,buff1)+1
  datapt=createBuff(datapt,data.value,buff2)+1
  datapt=createBuff(datapt,data.value,buff3)+1
  
  gwstruct1={"h":buff1,"t":[st0,sdt]}
  st0=st0+sdt*BUFF_SZ

  

  gwstruct2={"h":buff2,"t":[st0,sdt]}
  st0=st0+sdt*BUFF_SZ

  gwstruct3={"h":buff2,"t":[st0,sdt]}

  gwlist=[gwstruct1,gwstruct2,gwstruct3]

  asyncio.run(parallPost(gwlist))

  f=open("lastrd.txt","w")
  f.write(str(datapt))
  f.close()
  
  
  










if __name__=="__main__":
     main()





