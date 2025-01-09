package fbufstat

import (
	"encoding/json"
	"sync"
)

type Bufstat struct {
	N_itm     int `json:"n_itm"`
	Buff_size int `json:"buff_size"`
}

var instance *Bufstat
var once sync.Once
var mu sync.Mutex
var nu sync.Mutex

func GetInst() *Bufstat {

	once.Do(func() {

		instance = &Bufstat{0, 0}

	})

	return instance
}

func (Class *Bufstat) SetStat(buf []byte) {

	var Ftable Bufstat

	json.Unmarshal(buf, &Ftable)

	Class.N_itm = Ftable.N_itm
	Class.Buff_size = Ftable.Buff_size

}

func (Class *Bufstat) GetObj() *Bufstat {

	return Class

}

func (Class *Bufstat) UpdateCnt() {

	mu.Lock()
	Class.N_itm += 1
	mu.Unlock()

}

func (Class *Bufstat) CancCnt() {

	mu.Lock()
	Class.N_itm -= 1
	mu.Unlock()

}

func (Class Bufstat) GetJSONObj() []byte {

	bt_arr, _ := json.Marshal(Class)

	return bt_arr

}

func (Class *Bufstat) UpdateSize(buf int) {

	mu.Lock()
	Class.Buff_size += buf
	mu.Unlock()

}

func (Class *Bufstat) CancSize(buf int) {

	mu.Lock()
	Class.Buff_size -= buf
	mu.Unlock()

}

func (Class *Bufstat) UpdateStMux(bufs int) {

	mu.Lock()
	Class.N_itm += 1
	Class.Buff_size += bufs
	mu.Unlock()

}

func (Class *Bufstat) UpdateSt(bufs int) {

	Class.N_itm += 1
	Class.Buff_size += bufs

}
