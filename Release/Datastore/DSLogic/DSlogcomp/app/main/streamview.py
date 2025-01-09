from flask import request,Response
from app.main.main import main_bp
from .. import AppStatus
from .. import DBConn




@main_bp.route('/stream',methods=['POST'])
def pushStream():
     
     desc=DBConn.getDSDsc()
     state=desc['resp_msg']['dstatus']

     """"""
    
     if(AppStatus.isOp(state)):

      data=request.data

      #print('Incoming data',file=sys.stderr)
      #print(request.data,file=sys.stderr)

    

      output=DBConn.postDataStream(data)

      if (output['code']==201): 
        appresp=Response(str(output['resp_msg']),status=201,mimetype='text/plain') 
      """
      else:
       appresp=Response("Internal error!",status=500,mimetype='text/plain')   
      """
     else: 
      appresp=Response("Datastore frozen.",status=503,mimetype='text/plain')   

     return appresp

