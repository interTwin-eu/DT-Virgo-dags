from app.main.main import main_bp
from .. import DBConn


@main_bp.route('/dsdesc',methods=['GET'])
def getDsc():
 output=DBConn.getDSDsc()


 if (output['code']==200): 
        appresp=output['resp_msg']
      
     
 return appresp    