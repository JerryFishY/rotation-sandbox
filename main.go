package main

import (
	"flag"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

var (
	logN     = flag.Int("logN", 8, "logarithm of polynomial degree")
	test     = flag.String("test", "Decoder", "the module to test")
	level    = flag.Int("level", 6, "input level of the module")
	btpLevel = flag.Int("btpLevel", 15, "bootstrap level of the module, limited to Norm and Softmax")
	hidDim   = flag.Int("hidDim", 32, "hidden dimension of the model")
	expDim   = flag.Int("expDim", 64, "expanded hidden dimension of the model")
	seqLen   = flag.Int("seqLen", 29, "input sequence length")
	numHeads = flag.Int("numHeads", 2, "number of heads")
	parallel  = flag.Bool("parallel", false, "use parallel computing or not")
)

func main() {
	flag.Parse()

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            *logN,
		LogQ:            []int{53, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41},
		LogP:            []int{61, 61, 61, 61},
		LogDefaultScale: 41,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		panic(err)
	}
	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN: logN,
		LogP: []int{61, 61, 61, 61},
		Xs:   params.Xs(),
	}
	llama, helper, size, opeval := PrepareContext(params, btpParametersLit)

	fmt.Print("Initialization finished!\n")
	x := helper.ctGen(1)[0]
	xDec := make([]complex128, x.Slots())
	helper.encoder.Decode(helper.decryptor.DecryptNew(x), xDec)
	xMsg := make([]complex128, size.hidDim)
	for i := 0; i < size.hidDim; i++ {
		xMsg[i] = xDec[i*x.Slots()/size.hidDim]
	}
	switch *test {
	case "QKV":
		helper.PrepareWeights(size, []string{"q", "k", "v"}, llama)
		qCt, _, _ := llama.QKV(x)
		qPad := llama.helper.Dec(qCt, 0)
		q := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			q[i] = qPad[i*x.Slots()/size.hidDim]
		}
		qMsg := llama.LinearMsg(xMsg, llama.wMsg["q"])
		llama.helper.MSE(q, qMsg)
	case "RoPE":
		helper.PrepareWeights(size, []string{"RoPE"}, llama)
		xCt, _ := llama.RoPE(x, x)
		xPad := llama.helper.Dec(xCt, 0)
		xOut := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			xOut[i] = xPad[i*x.Slots()/size.hidDim]
		}
		xMsgRot, _ := llama.RoPEMsg(xMsg, xMsg)
		llama.helper.MSE(xOut, xMsgRot)
	case "Cache":
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		llama.Cache(x, x)
	case "QK_T":
		helper.PrepareCache(size, []string{"k"}, llama)
		sCt := llama.QK_T(x)
		sMsg := llama.LinearMsg(xMsg, llama.cacheMsg["k"], 2)
		llama.helper.Dec(sCt, 10)
		fmt.Print(sMsg[:10], "\n")
	case "AttnV":
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		sCt := llama.QK_T(x)
		sMsg := llama.LinearMsg(xMsg, llama.cacheMsg["k"], 2)
		llama.helper.Dec(sCt, 128)
		fmt.Print(sMsg[:58], "\n")
		oCt := llama.AttnV(sCt)
		oMsg := llama.AttnVMsg(sMsg)
		llama.helper.Dec(oCt, 128)
		fmt.Print(oMsg[:32], "\n")
	case "Out":
		helper.PrepareWeights(size, []string{"out"}, llama)
		llama.Out(x)
	case "UpGate":
		helper.PrepareWeights(size, []string{"up", "gate"}, llama)
		upCt, _ := llama.UpGate(x)
		upPad := llama.helper.Dec(upCt, 0)
		up := make([]complex128, size.expDim)
		for i := 0; i < size.expDim; i++ {
			up[i] = upPad[i*x.Slots()/size.expDim]
		}
		upMsg := llama.LinearMsg(xMsg, llama.wMsg["up"])
		llama.helper.MSE(up, upMsg)
	case "Down":
		helper.PrepareWeights(size, []string{"up", "gate"}, llama)
		helper.PrepareWeights(size, []string{"down"}, llama)
		upCt, _ := llama.UpGate(x)
		downCt := llama.Down(upCt)
		downPad := llama.helper.Dec(downCt, 0)
		down := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			down[i] = downPad[i*x.Slots()/size.hidDim]
		}
		upMsg := llama.LinearMsg(xMsg, llama.wMsg["up"])
		downMsg := llama.LinearMsg(upMsg, llama.wMsg["down"])
		llama.helper.MSE(down, downMsg)
	case "SiLU":
		y_pt := llama.SiLUPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.SiLU(x), 0)
		llama.helper.MSE(y, y_pt)
	case "Softmax":
		y_pt := llama.SoftmaxPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.Softmax(x, *btpLevel, 0), 0)
		llama.helper.MSE(y, y_pt)
	case "Norm":
		y_pt := llama.NormPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.Norm(x, *btpLevel), 0)
		llama.helper.MSE(y, y_pt)
	case "NormThor":
		y_pt := llama.NormPlaintext(llama.helper.Dec(x, 0))
		y := llama.helper.Dec(llama.NormThor(x, 0), 0)
		llama.helper.MSE(y, y_pt)
	case "Argmax":
		// fmt.Printf("Plaintext argmax: %d\n", llama.ArgmaxPlaintext(llama.helper.Dec(x, 0)))
		pt := llama.ArgmaxPlaintext(llama.helper.Dec(x, 0))
		fmt.Print("Plaintext argmax: ")
		for _, val := range(pt) {
			fmt.Printf("%d+%di; ", val % 16, val / 16)
		}
		fmt.Print("\n")
		llama.helper.Dec(llama.Argmax(x), 8)
	case "CtMult":
		opeval.ctCtMult()
	case "Ops":
		opeval.EvaluateAll()
	case "Decoder":
		fmt.Printf("Preparing model...\n")
		helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
		helper.PrepareCache(size, []string{"k", "v"}, llama)
		fmt.Printf("Preparation finished!\nEvaluating one decoder...\n")
		outCt := llama.DecoderThor(x)
		outPad := llama.helper.Dec(outCt, 0)
		out := make([]complex128, size.hidDim)
		for i := 0; i < size.hidDim; i++ {
			out[i] = outPad[i*x.Slots()/size.hidDim]
		}
		outMsg := llama.DecoderMsg(xMsg)
		llama.helper.MSE(out, outMsg)
	case "Model":
		fmt.Printf("Preparing model...\n")
		helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
		helper.PrepareCache(size, []string{"k", "v", "mask"}, llama)
		fmt.Printf("Preparation finished!\nEvaluating End-to-end Inference!\n")
		llama.Model(x)
	default:
		fmt.Print("Please specify the module to evaluate.")
	}
}
