package simplelogger

import (
	"errors"
	"os"

	"github.com/rs/zerolog"
)

type LogWrite struct {
	Level   string `json:"level"`
	Id      int    `json:"Id"`
	Size    int    `json:"size"`
	Name    string `json:"name"`
	Time    string `json:"time"`
	Message string `json:"message"`
}

func LogWriteFile(dest string, filename string, i int, size int, fname string) error {

	file, err := os.OpenFile(
		dest+filename,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0666,
	)
	if err != nil {
		return err
	}

	defer file.Close()

	logger := zerolog.New(file).With().Timestamp().Logger()

	logger.Info().Int("id", i).Int("size", size).Str("name", fname).Msg("Write file")

	return err

}

func LogPanic(main_err string, descr string) {

	err := errors.New(main_err)

	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	logger.Panic().Err(err).Msg(descr)

}

func LogGreet(greeter string) {

	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()
	logger.Info().Msg(greeter)

}

func LogInf(dest string, filename string, info string) error {

	file, err := os.OpenFile(
		dest+filename,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0666,
	)
	if err != nil {
		return err
	}

	defer file.Close()

	logger := zerolog.New(file).With().Timestamp().Logger()

	logger.Info().Msg(info)

	return err

}
