package main

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type testExpParameters struct {
	LogN            int
	LogQ            []int
	LogP            []int
	LogDefaultScale int
	Xs              ring.Ternary
}

func getDefaultExpTestParams() testExpParameters {
	return testExpParameters{
		LogN:            12,
		LogQ:            []int{52, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192},
	}
}

type testExpContext struct {
	params *ckks.Parameters
	llama  *LlamaInference
	helper *TestHelper
}

func setupExpTestContext(t *testing.T, params testExpParameters) *testExpContext {
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            params.LogN,
		LogQ:            params.LogQ,
		LogP:            params.LogP,
		LogDefaultScale: params.LogDefaultScale,
		Xs:              params.Xs,
	})
	require.NoError(t, err)

	btpParamsLit := bootstrapping.ParametersLiteral{
		LogN: &params.LogN,
		LogP: []int{61, 61, 61, 61},
		Xs:   ckksParams.Xs(),
	}

	llama, helper, _, _ := PrepareContext(ckksParams, btpParamsLit)

	return &testExpContext{
		params: &ckksParams,
		llama:  llama,
		helper: helper,
	}
}

func TestCExp8xSmall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupExpTestContext(t, getDefaultExpTestParams())

	// Generate test values in range [-27.2493, 21.72692]
	// Pre-scale by 1/32 (upstream softmax will handle this)
	n := tc.params.MaxSlots()
	xValues := make([]float64, n)
	for i := 0; i < n; i++ {
		xValues[i] = sampling.RandFloat64(-25, 20) / 32.0 // Pre-scale by 1/32
	}

	// Encrypt
	ptX := ckks.NewPlaintext(*tc.params, 10)
	tc.helper.encoder.Encode(xValues, ptX)
	ctX, err := tc.helper.encryptor.EncryptNew(ptX)
	require.NoError(t, err)

	// Compute ciphertext result (0 squaring iterations by default)
	start := time.Now()
	ctY := tc.llama.cExp8xSmall(ctX, 0)
	elapsed := time.Since(start)

	// Decrypt
	msgY := make([]complex128, ctY.Slots())
	tc.helper.encoder.Decode(tc.helper.decryptor.DecryptNew(ctY), msgY)

	// Compute plaintext reference
	ptXComplex := make([]complex128, n)
	for i, v := range xValues {
		ptXComplex[i] = complex(v, 0)
	}
	yExpected := tc.llama.cExp8xSmallPlaintext(ptXComplex)

	// Compare
	var maxErr float64
	for i := 0; i < n; i++ {
		diff := real(msgY[i]) - yExpected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxErr {
			maxErr = diff
		}
	}

	mse := math.Sqrt(maxErr * maxErr)
	precision := -math.Log2(mse)

	t.Logf("Precision: %.2f bits", precision)
	t.Logf("Time: %v", elapsed)
	t.Logf("Input level: %d, Output level: %d", ctX.Level(), ctY.Level())

	require.Greater(t, precision, 8.0, "precision %.2f bits is less than required 8.0 bits", precision)
	require.Less(t, elapsed.Seconds(), 60.0, "cExp8xSmall took too long: %v", elapsed)
}

func TestCExp8xLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupExpTestContext(t, getDefaultExpTestParams())

	// Generate test values in range [-70, 70]
	// Pre-scale by 1/64 (upstream softmax will handle this)
	n := 128
	xValues := make([]float64, n)
	for i := 0; i < n; i++ {
		xValues[i] = sampling.RandFloat64(-60, 60) / 64.0 // Pre-scale by 1/64
	}

	// Encrypt
	ptX := ckks.NewPlaintext(*tc.params, 10)
	tc.helper.encoder.Encode(xValues, ptX)
	ctX, err := tc.helper.encryptor.EncryptNew(ptX)
	require.NoError(t, err)

	// Compute ciphertext result (0 squaring iterations by default)
	start := time.Now()
	ctY := tc.llama.cExp8xLarge(ctX, 0)
	elapsed := time.Since(start)

	// Decrypt
	msgY := make([]complex128, ctY.Slots())
	tc.helper.encoder.Decode(tc.helper.decryptor.DecryptNew(ctY), msgY)

	// Compute plaintext reference
	ptXComplex := make([]complex128, n)
	for i, v := range xValues {
		ptXComplex[i] = complex(v, 0)
	}
	yExpected := tc.llama.cExp8xLargePlaintext(ptXComplex)

	// Compare
	var maxErr float64
	for i := 0; i < n; i++ {
		diff := real(msgY[i]) - yExpected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxErr {
			maxErr = diff
		}
	}

	mse := math.Sqrt(maxErr * maxErr)
	precision := -math.Log2(mse)

	t.Logf("Precision: %.2f bits", precision)
	t.Logf("Time: %v", elapsed)
	t.Logf("Input level: %d, Output level: %d", ctX.Level(), ctY.Level())

	require.Greater(t, precision, 8.0, "precision %.2f bits is less than required 8.0 bits", precision)
	require.Less(t, elapsed.Seconds(), 60.0, "cExp8xLarge took too long: %v", elapsed)
}
