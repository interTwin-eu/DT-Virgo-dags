from app.main.main import main_bp
from .. import DBConn


@main_bp.route('/dumpflog',methods=['GET'])
def getFlog():
     
     
     output=DBConn.getLogDump()

     #print(output['resp_msg'],file=sys.stderr)

     

     if (output['code']==200): 
       appresp=output['resp_msg']

       """
     else:
       appresp=Response("Internal error!",status=500,mimetype='text/plain')   
     """
     

     return appresp