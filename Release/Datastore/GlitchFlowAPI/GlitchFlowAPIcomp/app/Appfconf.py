import os

"""

ENV. TO BE DEFINED INSIDE THE CONTAINER IMG

ENDP_STAT
ENDP_DUMP
ENDP_SEND
DB_BURL
"""

env_stat=os.environ['DS_STAT']

env_send=os.environ['DS_SEND']
env_freeze=os.environ['DS_FREEZE']
env_burl=os.environ['DS_BURL']
env_clean=os.environ['DS_FLUSH']
env_desc=os.environ['DS_DESC']
env_dumpbuf=os.environ['DS_DUMPF']
env_max_size=os.environ['MAX_SIZE']




