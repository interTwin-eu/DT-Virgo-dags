package neterr

const CdErrIO = "ERRIO"

const BodyFIO = "Fs error detected"

type ErrResp struct {
	Code string `json:"code"`
	Body string `json:"body"`
}

func New(codstr string, bodystr string) ErrResp {

	Resp := ErrResp{codstr, bodystr}

	return Resp

}
