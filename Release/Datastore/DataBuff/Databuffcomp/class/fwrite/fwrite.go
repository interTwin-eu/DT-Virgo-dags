package fwrite

/*
The scope of this class is to provide methods for read/write streams
of bytes. Every operation i/o operation should be atomical

i.e. I receive a stream from endpoints. Write a new file locally
i.e Reading single file
*/

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"

	"github.com/google/renameio"
)

type FList struct {
	NMList []string `json:"nmlist"`
}

// If the func literal starts with an uppercase letter
// the function will be public
func StructToByte(buf any) []byte {

	bt_arr, _ := json.Marshal(buf)

	return bt_arr

}

func PrintStToF(dest string, filename string, mod fs.FileMode, DtStr any) (int, error) {

	var code error
	var fsize int

	buf := StructToByte(DtStr)
	fsize = len(buf)

	//0644 read-only mod for other users.
	//0666 THE NUMBER OF THE BEAST! Read and write for all!
	//Remember WriteFile return nil with success.
	code = os.WriteFile(dest+filename, buf, mod)

	return fsize, code

}

func PrintStrmToF(dest string, filename string, mod fs.FileMode, DtStr []byte) error {

	//0644 read-only mod for other users.
	//0666  Read and write for all
	//Remember WriteFile return nil with success.
	code := os.WriteFile(dest+filename, DtStr, mod)

	return code

}

func FtoStrm(dest string, filename string) ([]byte, error) {

	var code error
	var bt_arr []byte

	bt_arr, code = os.ReadFile(dest + filename)

	return bt_arr, code

}

func UnFtoStrm(dest string, filename string) ([]byte, error) {

	var bt_arr []byte
	var err error

	bt_arr, err = os.ReadFile(dest + filename)

	return bt_arr, err

}

func RFbyLine(dest string, filename string) ([]string, int) {

	var buf []string
	i := 0
	var code error

	file, code := os.Open(dest + filename)

	if code == nil {
		scanner := bufio.NewScanner(file)

		for scanner.Scan() {
			line := scanner.Text()
			buf = append(buf, line)
			i++

		}

		if scanner.Err() != nil {

			i = 0
		}

		file.Close()

	}

	return buf, i

}

func AtmWrtJs(dest string, name string, data any) (int, error) {

	var cbuf []byte

	cbuf, _ = json.Marshal(data)

	t, err := renameio.TempFile("", dest+name)
	if err != nil {
		return 0, err
	}
	defer t.Cleanup()

	if count, err := fmt.Fprintf(t, "%s", cbuf); err != nil {
		return 0, err
	} else {
		return count, t.CloseAtomicallyReplace()
	}

}
