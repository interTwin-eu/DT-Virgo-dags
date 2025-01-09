package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"base.url/class/appmodel"
	"base.url/class/appstr"
	"base.url/class/dsstat"
	"base.url/class/envdef"
	"base.url/class/fbufstat"
	"base.url/class/fwrite"
	"base.url/class/simplelogger"
	"github.com/gin-gonic/gin"
)

func main() {

	var nch int
	StatDesc := fbufstat.GetInst()
	DSDesc := dsstat.GetInst()

	nch, _ = strconv.Atoi(envdef.Chlen)

	wrtch := make(chan int, nch)
	msgch := make(chan int)

	if initstat(envdef.Baseadm, envdef.Baseadmn, envdef.DStnm, StatDesc, DSDesc) != nil {

		simplelogger.LogPanic("FATAL ERROR", "FS ERROR")

	}

	simplelogger.LogGreet("DB ready")

	App := appstr.ConcrH{}

	gin.SetMode(gin.ReleaseMode)

	router := gin.Default()

	initapp(router, App, StatDesc, DSDesc, wrtch)

	srv := &http.Server{Addr: envdef.Basesrvurl, Handler: router}

	go worker(wrtch, msgch, StatDesc)

	go func() {
		// service connections
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			simplelogger.LogPanic("FATAL ERROR", "NETWORK ERROR")
		}
	}()

	quitch := make(chan os.Signal, 1)

	signal.Notify(quitch, syscall.SIGINT, syscall.SIGTERM)
	//main blocks until a signal is received
	<-quitch

	simplelogger.LogGreet("Init shutdown...")

	ct, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ct); err != nil {
		simplelogger.LogPanic("FORCING SHUTDOWN!", "NETWORK ERROR")
	}

	// catching ct.Done(). timeout of 5 seconds.
	simplelogger.LogGreet("Timeout of 5 seconds...")
	<-ct.Done()

	simplelogger.LogGreet("Server exiting.")
	msgch <- 0
	simplelogger.LogGreet("Timeout of 10 seconds...")
	time.Sleep(10 * time.Second)
	simplelogger.LogGreet("Databuff closing...")

}

func worker(Bch chan int, Mch chan int, Stpt *fbufstat.Bufstat) {

	var buf int
	var err error

	for {

		select {

		case buf = <-Bch:

			Stpt.UpdateSt(buf)
			_, err = fwrite.AtmWrtJs(envdef.Baseadm, envdef.Baseadmn, Stpt)

			if err != nil {

				simplelogger.LogPanic("FATAL ERROR", "FS ERROR")

			}

		case <-Mch:

			simplelogger.LogGreet("Main worker thread stopping...")
			return

		}

	}

}

func initstat(rtdir string, rtfname string, descfname string, Buf *fbufstat.Bufstat, Dsd *dsstat.DSstat) error {
	var err, derr, initerr error
	var cbuf, dbuf []byte

	initerr = nil

	cbuf, err = fwrite.UnFtoStrm(rtdir, rtfname)
	dbuf, derr = fwrite.UnFtoStrm(rtdir, descfname)

	if (err == nil) && (derr == nil) {

		Buf.SetStat(cbuf)
		Dsd.SetStat(dbuf)
	} else {

		initerr = errors.New("FS ERR")
	}

	return initerr
}

func PostWho(Desc *dsstat.DSstat) gin.HandlerFunc {

	return func(ctx *gin.Context) {

		var buf dsstat.Idt

		ctx.BindJSON(&buf)

		Desc.User = buf.User
		Desc.Token = buf.Token

		ctx.Next()

	}
}

func initapp(SrvPt *gin.Engine, AppRts appmodel.AbstrApp, Buf *fbufstat.Bufstat, Desc *dsstat.DSstat, Chnl chan int) {

	SrvPt.GET("/stat", AppRts.GetStat(Buf))
	SrvPt.GET("/dumpLogF", AppRts.GetLogF())
	SrvPt.GET("/dstat", AppRts.GetDSdsc(Desc))
	SrvPt.POST("/sendF", AppRts.PostFile(Chnl))
	SrvPt.POST("/upddsc", PostWho(Desc), AppRts.PostDsc(Desc))
	SrvPt.POST("/cleanall", AppRts.PostClean(Buf, Desc))
	SrvPt.GET("/dumpF", AppRts.GetFLst())

}
