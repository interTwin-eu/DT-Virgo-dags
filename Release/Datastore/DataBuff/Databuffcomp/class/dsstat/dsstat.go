package dsstat

import (
	"encoding/json"
	"sync"
)

type DSstat struct {
	Dstatus string `json:"dstatus"`
	User    string `json:"user"`

	Token string `json:"token"`
}

type Idt struct {
	User string `json:"user"`

	Token string `json:"token"`
}

var instance *DSstat
var once sync.Once
var mu sync.Mutex

const opstr = "OPERATIONAL"
const frzstr = "FROZEN"

func GetInst() *DSstat {

	once.Do(func() {

		instance = &DSstat{opstr, "databuf", "databuf"}

	})

	return instance
}

func (Class *DSstat) SetToFrz() {

	mu.Lock()
	Class.Dstatus = frzstr
	mu.Unlock()

}

func (Class *DSstat) SetToOp() {

	mu.Lock()
	Class.Dstatus = opstr
	mu.Unlock()

}

func (Class *DSstat) SetUsr(buf string) {

	mu.Lock()
	Class.User = buf
	mu.Unlock()

}

func (Class *DSstat) SetTok(buf string) {

	mu.Lock()
	Class.Token = buf
	mu.Unlock()

}

func (Class *DSstat) SetStat(buf []byte) {

	var Ftable DSstat

	json.Unmarshal(buf, &Ftable)

	Class.Dstatus = Ftable.Dstatus
	Class.Token = Ftable.Token
	Class.User = Ftable.User

}
