from flask import render_template
from app.main.main import main_bp
from .. import AppStatus
from .. import DBConn





@main_bp.route('/')
def index():
    
    output=DBConn.getBufStat()
    output_desc=DBConn.getDSDsc()

    output_f=DBConn.getDumpF()

     #print(output['resp_msg'],file=sys.stderr)

    if (output['code']==200): 
       

       appresp=render_template(
          'index.html',size=output['resp_msg']['buff_size']
          ,state=output_desc['resp_msg']['dstatus'],
          nitm=output['resp_msg']['n_itm'],
          user=output_desc['resp_msg']['user'],
          token=output_desc['resp_msg']['token'],
          files=output_f['resp_msg']['nmlist'])
    
    
    return appresp