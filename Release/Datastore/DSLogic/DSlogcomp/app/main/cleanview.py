from flask import request,Response
from app.main.main import main_bp
from .. import DBConn

@main_bp.route('/flushbuf',methods=['POST'])
def flushBuf():

 data=request.data

      
    
 output=DBConn.postFlush(data)

 if (output['code']==201): 
   appresp=Response(str(output['resp_msg']),status=201,mimetype='application/json') 
      
 else: 
      appresp=Response("Broken FS",status=500,mimetype='application/json')   

 return appresp

     
