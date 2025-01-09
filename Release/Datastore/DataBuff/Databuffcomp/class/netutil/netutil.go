package netutil

import (
	"base.url/class/fbufstat"
	"github.com/gin-gonic/gin"
)

func SendJSON(Buf fbufstat.Bufstat, Hnd *gin.Context) {

	Hnd.JSON(200, Buf)

}
