#!/bin/bash

###va lanciato con source

<<comment


GIN CALLS

router.GET("/stat", getStat(&StatDesc))
router.GET("/dumpLogF", getLogF())

router.POST("/sendF", postFile(&StatDesc))   

comment


export ENDP_STAT=stat
export ENDP_DUMP=dumpLogF
export ENDP_DUMPF=dumpF

export ENDP_SEND=sendF
export DB_BURL=http://localhost:8080/
export ENDP_DESC=dstat
export ENDP_UPD_DESC=upddsc
export ENDP_CLEAN=cleanall
export MAX_SIZE=2000




