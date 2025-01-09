class AppDesc:
    """
    def __init__(self,size):
        self.buf_state='OPERATIONAL'
        self.max_buf_size=int(size)
    """
    
    def initDesc(self,size):
        
        self.max_buf_size=int(size)
        self.frz_state='FROZEN'
        self.op_state='OPERATIONAL'

    
    
    def isFreeze(self,curr_size):

        code=0

        if(curr_size>=self.max_buf_size):
            code=1

        return code  
    
    def isOp(self,status): 
        code=0

        if(status==self.op_state):
            code=1

        return code    

              
        