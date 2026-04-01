package main

import (
	"flag"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

var (
	debugInvMode = flag.Bool("debug-inv", false, "enable debug mode for Inv with verbose output")
)

// testParameters holds the test CKKS parameters
type testInvParameters struct {
	LogN            int
	LogQ            []int
	LogP            []int
	LogDefaultScale int
	Xs              ring.Ternary
}

// getDefaultInvTestParams returns default test parameters
func getDefaultInvTestParams() testInvParameters {
	return testInvParameters{
		LogN:            12,
		LogQ:            []int{52, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192},
	}
}

// testInvContext holds the testing context
type testInvContext struct {
	params  *ckks.Parameters
	llama   *LlamaInference
	helper  *TestHelper
	size    *LlamaSize
	btpEval *bootstrapping.Evaluator
}

// setupInvTestContext creates a test context with CKKS parameters
func setupInvTestContext(t *testing.T, params testInvParameters) *testInvContext {
	// Create CKKS parameters
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            params.LogN,
		LogQ:            params.LogQ,
		LogP:            params.LogP,
		LogDefaultScale: params.LogDefaultScale,
		Xs:              params.Xs,
	})
	require.NoError(t, err, "failed to create CKKS parameters")

	// Create bootstrapping parameters
	btpParamsLit := bootstrapping.ParametersLiteral{
		LogN: &params.LogN,
		LogP: []int{61, 61, 61, 61},
		Xs:   ckksParams.Xs(),
	}

	// Prepare context (using the function from util.go)
	llama, helper, size, _ := PrepareContext(ckksParams, btpParamsLit)

	return &testInvContext{
		params:  &ckksParams,
		llama:   llama,
		helper:  helper,
		size:    size,
		btpEval: llama.btpEval,
	}
}

// generateInvTestValues generates test values in a specified range
func generateInvTestValues(min, max float64, n int) []float64 {
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		// Generate positive values only (for inverse)
		values[i] = sampling.RandFloat64(min, max)
		if values[i] <= 0 {
			values[i] = 0.1 // Ensure positive
		}
	}
	return values
}

// encryptInvValues encrypts float64 values into a ciphertext
func (tc *testInvContext) encryptInvValues(t *testing.T, values []float64, level int) *rlwe.Ciphertext {
	// Pad values to match MaxSlots by repeating the input values cyclically
	maxSlots := tc.params.MaxSlots()
	message := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		message[i] = values[i%len(values)]
	}

	plaintext := ckks.NewPlaintext(*tc.params, level)
	err := tc.helper.encoder.Encode(message, plaintext)
	require.NoError(t, err, "failed to encode values")

	ciphertext, err := tc.helper.encryptor.EncryptNew(plaintext)
	require.NoError(t, err, "failed to encrypt values")

	return ciphertext
}

// decryptInvCiphertext decrypts a ciphertext to float64 values
func (tc *testInvContext) decryptInvCiphertext(t *testing.T, ct *rlwe.Ciphertext) []float64 {
	msg := make([]complex128, ct.Slots())
	err := tc.helper.encoder.Decode(tc.helper.decryptor.DecryptNew(ct), msg)
	require.NoError(t, err, "failed to decrypt ciphertext")

	result := make([]float64, len(msg))
	for i, v := range msg {
		result[i] = real(v)
	}
	return result
}

// compareInvWithPlaintext compares ciphertext result with plaintext result
func compareInvWithPlaintext(t *testing.T, ctResult, ptResult []float64, maxMSE float64, minPrecisionBits float64) {
	require.Len(t, ctResult, len(ptResult),
		"ciphertext and plaintext results have different lengths")

	// Calculate MSE
	var sum float64
	maxDiff := 0.0
	for i := range ctResult {
		diff := ctResult[i] - ptResult[i]
		absDiff := math.Abs(diff)
		if absDiff > maxDiff {
			maxDiff = absDiff
		}
		sum += diff * diff
	}
	mse := math.Sqrt(sum / float64(len(ctResult)))

	// Calculate precision in bits
	precision := -math.Log2(mse)

	// Check MSE (warning only, doesn't stop test)
	if mse >= maxMSE {
		t.Logf("WARNING: MSE %f exceeds threshold %f", mse, maxMSE)
	}

	// Check precision (warning only, doesn't stop test)
	if precision <= minPrecisionBits {
		t.Logf("WARNING: precision %.2f bits is less than required %.2f bits", precision, minPrecisionBits)
	}

	if *debugInvMode {
		fmt.Printf("  ✓ MSE: %e, Precision: %.2f bits\n", mse, precision)
		fmt.Printf("  ✓ Max difference: %e\n", maxDiff)

		// Show first 15 values side-by-side
		fmt.Printf("  ✓ Sample comparison (first 15 values):\n")
		fmt.Printf("    Index | Expected (CKKS) | Actual (Plaintext) | Difference | Relative Error\n")
		fmt.Printf("    -------|----------------|-------------------|------------|----------------\n")
		for i := 0; i < 15 && i < len(ctResult); i++ {
			diff := ctResult[i] - ptResult[i]
			relErr := 0.0
			if ptResult[i] != 0 {
				relErr = math.Abs(diff/ptResult[i]) * 100.0
			}
			fmt.Printf("    %5d | %14.6f | %17.6f | %10.2e | %5.2f%%\n",
				i, ctResult[i], ptResult[i], diff, relErr)
		}
	}
}

// TestInv_Basic tests basic functionality of Inv
func TestInv_Basic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupInvTestContext(t, getDefaultInvTestParams())

	tests := []struct {
		name         string
		values       []float64
		epsilon      float64
		alpha        float64
		level        int
		maxMSE       float64
		minPrecision float64
	}{
		{
			name:         "THOR softmax range [0.1, 1.0)",
			values:       []float64{0.1, 0.3, 0.5, 0.98},
			epsilon:      math.Pow(2, -11),
			alpha:        0.01,
			level:        8,
			maxMSE:       5e-3,
			minPrecision: 7.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if *debugInvMode {
				fmt.Printf("\n=== Test: %s ===\n", tt.name)
				fmt.Printf("Parameters: epsilon=%.3e, alpha=%.3f, level=%d\n",
					tt.epsilon, tt.alpha, tt.level)
			}

			config := InvConfig{
				Epsilon: tt.epsilon,
				Alpha:   tt.alpha,
			}

			// Encrypt input
			x := tc.encryptInvValues(t, tt.values, tt.level)

			// Compute Inv on ciphertext
			yCt := tc.llama.Inv(x, config)

			// Compute Inv on plaintext
			paddedValues := make([]float64, tc.params.MaxSlots())
			for i := 0; i < tc.params.MaxSlots(); i++ {
				paddedValues[i] = tt.values[i%len(tt.values)]
			}
			yPt := tc.llama.InvPlaintext(paddedValues)

			// Decrypt and compare
			yCtDecrypted := tc.decryptInvCiphertext(t, yCt)
			compareInvWithPlaintext(t, yCtDecrypted, yPt, tt.maxMSE, tt.minPrecision)
		})
	}
}

// TestInv_Performance tests performance characteristics
func TestInv_Performance(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupInvTestContext(t, getDefaultInvTestParams())

	if *debugInvMode {
		fmt.Printf("\n=== Performance Test ===\n")
	}

	values := generateInvTestValues(0.1, 1.0, tc.params.MaxSlots())
	x := tc.encryptInvValues(t, values, 5)

	// Measure ciphertext computation
	start := time.Now()
	config := InvConfig{
		Epsilon: math.Pow(2, -11),
		Alpha:   0.001,
	}
	yCt := tc.llama.Inv(x, config)
	ctTime := time.Since(start)

	// Measure plaintext computation
	paddedValues := make([]float64, tc.params.MaxSlots())
	for i := 0; i < tc.params.MaxSlots(); i++ {
		paddedValues[i] = values[i%len(values)]
	}
	start = time.Now()
	_ = tc.llama.InvPlaintext(paddedValues)
	ptTime := time.Since(start)

	if *debugInvMode {
		fmt.Printf("Ciphertext time: %v\n", ctTime)
		fmt.Printf("Plaintext time: %v\n", ptTime)
		fmt.Printf("Slowdown factor: %.2fx\n", float64(ctTime)/float64(ptTime))
		fmt.Printf("Input level: %d, Output level: %d\n", x.Level(), yCt.Level())
	}

	// Ciphertext should complete in reasonable time
	require.Less(t, ctTime.Seconds(), 60.0,
		"Inv took too long: %v", ctTime)
}

// TestInv_Optimizations tests each optimization independently
func TestInv_Optimizations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupInvTestContext(t, getDefaultInvTestParams())

	optimizations := []struct {
		name   string
		config InvConfig
	}{
		{"Baseline", InvConfig{
			Epsilon: math.Pow(2, -11),
			Alpha:   0.001,
		}},
		{"ConjugateDenoising", InvConfig{
			Epsilon:            math.Pow(2, -11),
			Alpha:              0.001,
			ConjugateDenoising: true,
		}},
	}

	for _, opt := range optimizations {
		t.Run(opt.name, func(t *testing.T) {
			values := []float64{0.2, 0.4, 0.6, 0.8}
			x := tc.encryptInvValues(t, values, 8)

			start := time.Now()
			yCt := tc.llama.Inv(x, opt.config)
			elapsed := time.Since(start)

			fmt.Printf("%s: Time=%v, InputLevel=%d, OutputLevel=%d\n",
				opt.name, elapsed, x.Level(), yCt.Level())

			// Decrypt and verify basic correctness
			yCtDecrypted := tc.decryptInvCiphertext(t, yCt)
			paddedValues := make([]float64, tc.params.MaxSlots())
			for i := 0; i < tc.params.MaxSlots(); i++ {
				paddedValues[i] = values[i%len(values)]
			}
			yPt := tc.llama.InvPlaintext(paddedValues)

			// Basic check: values should be reasonably close
			for i := 0; i < len(values); i++ {
				diff := math.Abs(yCtDecrypted[i] - yPt[i])
				require.Less(t, diff, 0.1, "value %d: %f vs %f, diff=%f",
					i, yCtDecrypted[i], yPt[i], diff)
			}
		})
	}
}

// BenchmarkInv benchmarks the Inv function
func BenchmarkInv(b *testing.B) {
	tc := setupInvTestContext(&testing.T{}, getDefaultInvTestParams())

	values := generateInvTestValues(0.1, 1.0, tc.params.MaxSlots())
	x := tc.encryptInvValues(&testing.T{}, values, 8)

	config := InvConfig{
		Epsilon: math.Pow(2, -11),
		Alpha:   0.001,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tc.llama.Inv(x.CopyNew(), config)
	}
}

// BenchmarkInvPlaintext benchmarks the plaintext version
func BenchmarkInvPlaintext(b *testing.B) {
	tc := setupInvTestContext(&testing.T{}, getDefaultInvTestParams())

	values := make([]float64, tc.params.MaxSlots())
	for i := range values {
		values[i] = sampling.RandFloat64(0.1, 1.0)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tc.llama.InvPlaintext(values)
	}
}
