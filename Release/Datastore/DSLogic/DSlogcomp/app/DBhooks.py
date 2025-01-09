from . DBCalls import RqCalls


"""
GIN CALLS

router.GET("/stat", getStat(&StatDesc))
router.GET("/dumpLogF", getLogF())

router.POST("/sendF", postFile(&StatDesc))   

"""

"""

Response struct

response={'code':req.status_code,'resp_msg':req.json()}
response={'code':req.status_code,'resp_msg':req.text}


"""

class DBhook:
    """
    def __init__(self,burl,stat,dump,send):
     self.Caller=RqCalls(burl)

     self.serviceurl=burl
     self.status=stat  
     self.log_dump=dump
     self.send_data=send
    """
    def getBufStat(self):
        
        buf_desc=self.Caller.getReq(self.Urls['stat'])

        return buf_desc
    
    def getLogDump(self):
       
       log_cont=self.Caller.getReq(self.Urls['dump'])

       return log_cont
    
    def getDumpF(self):
       
       f_cont=self.Caller.getReq(self.Urls['dumpf'])

       return f_cont
    
    def getDSDsc(self):
       
       buf_desc=buf_desc=self.Caller.getReq(self.Urls['desc'])

       return buf_desc

    def postDataStream(self,data_struct):


        #postJson expects a dict rapresenting a json object
        post_resp=self.Caller.postJson(self.Urls['send'],data_struct)

        return post_resp
    
    def postDsc(self,data_struct):


        #postJson expects a dict rapresenting a json object
        post_resp=self.Caller.postJson(self.Urls['upddesc'],data_struct)

        return post_resp
    
    def postFlush(self,data_struct):


        #postJson expects a dict rapresenting a json object
        post_resp=self.Caller.postJson(self.Urls['flush'],data_struct)

        return post_resp
    

    

    def connect(self,url_dict):
     self.Urls=url_dict
     self.Caller=RqCalls(self.Urls['burl'])

     
    

    






     







     