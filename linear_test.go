package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const stage1Seed int64 = 42

var (
	Stage1SeqLen   = 29
	Stage1HidDim   = 32
	Stage1ExpDim   = 64
	Stage1NumHeads = 2
)

const defaultMSETol = 1e-3

type stage1Context struct {
	params ckks.Parameters
	bpLit  bootstrapping.ParametersLiteral
	llama  *LlamaInference
	helper *TestHelper
	size   *LlamaSize
	slots  int
	hidStr int
	expStr int
	level  int
	logN   int
	tol    float64
	assert bool
}

func stage1MSE(got, want []complex128) (float64, error) {
	if len(got) != len(want) {
		return 0, fmt.Errorf("mse: length mismatch %d vs %d", len(got), len(want))
	}
	if len(got) == 0 {
		return 0, nil
	}
	var sum float64
	for i := range got {
		d := real(got[i]) - real(want[i])
		sum += d * d
	}
	return sum / float64(len(got)), nil
}

func precisionBitsFromMSE(mse float64) float64 {
	if mse <= 0 {
		return 64
	}
	return -math.Log2(math.Sqrt(mse))
}

func setupStage1Context(t *testing.T) *stage1Context {
	t.Helper()

	oldTest, oldLogN, oldLevel := *test, *logN, *level
	oldHidDim, oldExpDim, oldSeqLen, oldNumHeads := *hidDim, *expDim, *seqLen, *numHeads
	oldParallel := *parallel

	*test = "Decoder"
	*parallel = false
	*logN = 8
	*level = 6
	*hidDim = Stage1HidDim
	*expDim = Stage1ExpDim
	*seqLen = Stage1SeqLen
	*numHeads = Stage1NumHeads

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            *logN,
		LogQ:            []int{53, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41},
		LogP:            []int{61, 61, 61, 61},
		LogDefaultScale: 41,
		Xs:              ring.Ternary{H: 192},
	})
	if err != nil {
		t.Fatalf("failed to create CKKS parameters: %v", err)
	}
	bpLit := bootstrapping.ParametersLiteral{
		LogN: logN,
		LogP: []int{61, 61, 61, 61},
		Xs:   params.Xs(),
	}

	llama, helper, size, _ := PrepareContext(params, bpLit)
	helper.PrepareWeights(size, []string{"q", "k", "v", "out", "up", "gate", "down", "RoPE"}, llama)
	helper.PrepareCache(size, []string{"k", "v"}, llama)

	slots := params.MaxSlots()
	hidStr := slots / size.hidDim
	if hidStr <= 0 || hidStr*size.hidDim != slots {
		t.Fatalf("invalid hid stride: slots=%d hidDim=%d stride=%d", slots, size.hidDim, hidStr)
	}
	expStr := slots / size.expDim
	if expStr <= 0 || expStr*size.expDim != slots {
		t.Fatalf("invalid exp stride: slots=%d expDim=%d stride=%d", slots, size.expDim, expStr)
	}

	tol := defaultMSETol
	if v := os.Getenv("CACHEMIR_TOL"); v != "" {
		if parsed, e := strconv.ParseFloat(v, 64); e == nil && parsed > 0 {
			tol = parsed
		}
	}
	assertMode := os.Getenv("CACHEMIR_ASSERT") != "0"

	*test, *logN, *level = oldTest, oldLogN, oldLevel
	*hidDim, *expDim, *seqLen, *numHeads = oldHidDim, oldExpDim, oldSeqLen, oldNumHeads
	*parallel = oldParallel

	return &stage1Context{
		params: params,
		bpLit:  bpLit,
		llama:  llama,
		helper: helper,
		size:   size,
		slots:  slots,
		hidStr: hidStr,
		expStr: expStr,
		level:  6,
		logN:   8,
		tol:    tol,
		assert: assertMode,
	}
}

func makeStage1Input(t *testing.T, c *stage1Context, trial int, dim, stride int) (*rlwe.Ciphertext, []complex128) {
	t.Helper()
	rng := rand.New(rand.NewSource(stage1Seed + int64(trial)))
	msgSlots := make([]float64, c.slots)
	for i := range msgSlots {
		if i%stride == 0 {
			msgSlots[i] = -10 + 20*rng.Float64()
		}
	}
	pt := ckks.NewPlaintext(c.params, c.level)
	if err := c.helper.encoder.Encode(msgSlots, pt); err != nil {
		t.Fatalf("encode plaintext failed: %v", err)
	}
	xCt, err := c.helper.encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}
	xMsg := make([]complex128, dim)
	for i := range dim {
		xMsg[i] = complex(msgSlots[i*stride], 0)
	}
	return xCt, xMsg
}

func takeByStride(decoded []complex128, outDim, stride int) []complex128 {
	out := make([]complex128, outDim)
	for i := 0; i < outDim; i++ {
		out[i] = decoded[i*stride]
	}
	return out
}

func assertMSE(t *testing.T, c *stage1Context, block string, got, want []complex128) {
	t.Helper()
	mse, err := stage1MSE(got, want)
	if err != nil {
		t.Fatalf("%s mse failed: %v", block, err)
	}
	t.Logf("Stage1 block=%s | seqLen=%d hidDim=%d expDim=%d | MSE=%.3e | precision=%.2f bits",
		block, c.size.seqLen, c.size.hidDim, c.size.expDim, mse, precisionBitsFromMSE(mse))
	if c.assert && mse > c.tol {
		t.Fatalf("%s mse too large: got %.3e want <= %.3e", block, mse, c.tol)
	}
}

func TestStage1_LinearPipelineMSE(t *testing.T) {
	c := setupStage1Context(t)
	xCt, xMsg := makeStage1Input(t, c, 0, c.size.hidDim, c.hidStr)

	t.Run("QProjection", func(t *testing.T) {
		qCt, _, _ := c.llama.QKV(xCt.CopyNew())
		qPad := c.helper.Dec(qCt, 0)
		q := takeByStride(qPad, c.size.hidDim, c.hidStr)
		qMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["q"])
		assertMSE(t, c, "QProjection", q, qMsg)
	})

	t.Run("KProjection", func(t *testing.T) {
		_, kCt, _ := c.llama.QKV(xCt.CopyNew())
		kPad := c.helper.Dec(kCt, 0)
		k := takeByStride(kPad, c.size.hidDim, c.hidStr)
		kMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["k"])
		assertMSE(t, c, "KProjection", k, kMsg)
	})

	t.Run("VProjection", func(t *testing.T) {
		_, _, vCt := c.llama.QKV(xCt.CopyNew())
		vPad := c.helper.Dec(vCt, 0)
		v := takeByStride(vPad, c.size.hidDim, c.hidStr)
		vMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["v"])
		assertMSE(t, c, "VProjection", v, vMsg)
	})

	t.Run("OutProjection", func(t *testing.T) {
		oCt := c.llama.Out(xCt.CopyNew())
		oPad := c.helper.Dec(oCt, 0)
		o := takeByStride(oPad, c.size.hidDim, c.hidStr)
		oMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["out"])
		assertMSE(t, c, "OutProjection", o, oMsg)
	})

	t.Run("UpGateProjection", func(t *testing.T) {
		upCt, gateCt := c.llama.UpGate(xCt.CopyNew())
		upPad := c.helper.Dec(upCt, 0)
		gatePad := c.helper.Dec(gateCt, 0)
		up := takeByStride(upPad, c.size.expDim, c.expStr)
		gate := takeByStride(gatePad, c.size.expDim, c.expStr)
		upMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["up"])
		gateMsg := c.llama.LinearMsg(xMsg, c.llama.wMsg["gate"])
		assertMSE(t, c, "UpProjection", up, upMsg)
		assertMSE(t, c, "GateProjection", gate, gateMsg)
	})

}
