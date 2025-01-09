from app.main.main import main_bp
from .. import DBConn
import json


@main_bp.route('/dumpf',methods=['GET'])
def getFls():
     
     
     output=DBConn.getDumpF()

     #print(output['resp_msg'],file=sys.stderr)

     

     if (output['code']==200): 
       appresp=output['resp_msg']

       

       if(appresp ['nmlist']==None):
           appresp={"nmlist":[]}

         

      
     

     return appresp