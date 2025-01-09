import os
basedir=os.path.abspath(os.path.dirname(__file__))

"""

ENV. TO BE DEFINED INSIDE THE CONTAINER IMG

ENDP_STAT
ENDP_DUMP
ENDP_SEND
DB_BURL
ENDP_DESC
ENDP_UPD_DESC
"""
class Config:
   env_stat=os.environ['ENDP_STAT']
   env_desc=os.environ['ENDP_DESC']
   env_upd_desc=os.environ['ENDP_UPD_DESC']
   env_dump=os.environ['ENDP_DUMP']
   env_dumpf=os.environ['ENDP_DUMPF']
   env_send=os.environ['ENDP_SEND']
   env_burl=os.environ['DB_BURL']
   env_max_size=os.environ['MAX_SIZE']
   env_flush=os.environ['ENDP_CLEAN']

   env_dict=dict(burl=env_burl,
                 stat=env_stat,
                 dump=env_dump,
                 dumpf=env_dumpf,
                 send=env_send,
                 desc=env_desc,
                 upddesc=env_upd_desc,
                 flush=env_flush
                 )

   @staticmethod
   def init_app(app):
      pass
   
class DevConfig(Config):
   DEBUG=False

#class ProdConfig(Config):


config={'dev':DevConfig,'prod':None,'default':DevConfig}




