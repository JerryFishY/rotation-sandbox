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
	debugMode = flag.Bool("debug", false, "enable debug mode with verbose output")
)

// testParameters holds the test CKKS parameters
type testParameters struct {
	LogN            int
	LogQ            []int
	LogP            []int
	LogDefaultScale int
	Xs              ring.Ternary
}

// getDefaultTestParams returns default test parameters
func getDefaultTestParams() testParameters {
	return testParameters{
		LogN:            12,
		LogQ:            []int{52, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192},
	}
}

// testContext holds the testing context
type testContext struct {
	params  *ckks.Parameters
	llama   *LlamaInference
	helper  *TestHelper
	size    *LlamaSize
	btpEval *bootstrapping.Evaluator
}

// setupTestContext creates a test context with CKKS parameters
func setupTestContext(t *testing.T, params testParameters) *testContext {
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

	return &testContext{
		params:  &ckksParams,
		llama:   llama,
		helper:  helper,
		size:    size,
		btpEval: llama.btpEval,
	}
}

// generateTestValues generates test values in a specified range
func generateTestValues(min, max float64, n int) []float64 {
	values := make([]float64, n)
	for i := 0; i < n; i++ {
		// Generate positive values only (for inverse square root)
		values[i] = sampling.RandFloat64(min, max)
		if values[i] <= 0 {
			values[i] = 0.1 // Ensure positive
		}
	}
	return values
}

// encryptValues encrypts float64 values into a ciphertext
func (tc *testContext) encryptValues(t *testing.T, values []float64, level int) *rlwe.Ciphertext {
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

// decryptCiphertext decrypts a ciphertext to float64 values
func (tc *testContext) decryptCiphertext(t *testing.T, ct *rlwe.Ciphertext) []float64 {
	msg := make([]complex128, ct.Slots())
	err := tc.helper.encoder.Decode(tc.helper.decryptor.DecryptNew(ct), msg)
	require.NoError(t, err, "failed to decrypt ciphertext")

	result := make([]float64, len(msg))
	for i, v := range msg {
		result[i] = real(v)
	}
	return result
}

// compareWithPlaintext compares ciphertext result with plaintext result
func compareWithPlaintext(t *testing.T, ctResult, ptResult []float64, maxMSE float64, minPrecisionBits float64) {
	require.Len(t, ctResult, len(ptResult),
		"ciphertext and plaintext results have different lengths")

	// Calculate MSE
	var sum float64
	maxDiff := 0.0
	diffCount := make(map[int]int) // Track error distribution
	for i := range ctResult {
		diff := ctResult[i] - ptResult[i]
		absDiff := math.Abs(diff)
		if absDiff > maxDiff {
			maxDiff = absDiff
		}
		sum += diff * diff

		// Categorize errors
		errorBucket := int(math.Log10(absDiff + 1e-10))
		diffCount[errorBucket]++
	}
	mse := math.Sqrt(sum / float64(len(ctResult)))

	// Check MSE
	require.Less(t, mse, maxMSE,
		"MSE %f exceeds threshold %f", mse, maxMSE)

	// Calculate precision in bits
	precision := -math.Log2(mse)
	require.Greater(t, precision, minPrecisionBits,
		"precision %.2f bits is less than required %.2f bits", precision, minPrecisionBits)

	if *debugMode {
		fmt.Printf("  ✓ MSE: %e, Precision: %.2f bits\n", mse, precision)
		fmt.Printf("  ✓ Max difference: %e\n", maxDiff)

		// Show error distribution
		fmt.Printf("  ✓ Error distribution:\n")
		for order := -10; order <= 0; order++ {
			if count, ok := diffCount[order]; ok {
				fmt.Printf("    - 10^%d: %d values (%.1f%%)\n",
					order, count, 100.0*float64(count)/float64(len(ctResult)))
			}
		}

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

		// Show worst 5 mismatches
		fmt.Printf("  ✓ Worst 5 mismatches:\n")
		type mismatch struct {
			idx    int
			ctVal  float64
			ptVal  float64
			diff   float64
			relErr float64
		}
		worst := make([]mismatch, 0, len(ctResult))
		for i := range ctResult {
			diff := math.Abs(ctResult[i] - ptResult[i])
			relErr := 0.0
			if ptResult[i] != 0 {
				relErr = math.Abs((ctResult[i]-ptResult[i])/ptResult[i]) * 100.0
			}
			worst = append(worst, mismatch{i, ctResult[i], ptResult[i], diff, relErr})
		}
		// Sort by absolute difference
		for i := 0; i < len(worst)-1; i++ {
			for j := i + 1; j < len(worst); j++ {
				if worst[j].diff > worst[i].diff {
					worst[i], worst[j] = worst[j], worst[i]
				}
			}
		}
		nShow := 5
		if nShow > len(worst) {
			nShow = len(worst)
		}
		for i := 0; i < nShow; i++ {
			fmt.Printf("    [%d]: CKKS=%.6f, Plaintext=%.6f, Diff=%.2e, RelErr=%.2f%%\n",
				worst[i].idx, worst[i].ctVal, worst[i].ptVal, worst[i].diff, worst[i].relErr)
		}
	}
}

// debugInvSqrt runs InvSqrt with detailed debugging of each iteration
func debugInvSqrt(t *testing.T, tc *testContext, x *rlwe.Ciphertext, values []float64, epsilon, alpha float64) {
	fmt.Printf("\n╔════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║           DEBUG MODE: InvSqrt Iteration Analysis              ║\n")
	fmt.Printf("╚════════════════════════════════════════════════════════════════╝\n\n")

	// Prepare plaintext values for comparison by repeating cyclically
	paddedValues := make([]float64, tc.params.MaxSlots())
	for i := 0; i < tc.params.MaxSlots(); i++ {
		paddedValues[i] = values[i%len(values)]
	}

	// Expected plaintext results
	expected := make([]float64, len(paddedValues))
	for i, v := range paddedValues {
		expected[i] = 1.0 / math.Sqrt(v)
	}

	fmt.Printf("Input parameters:\n")
	fmt.Printf("  epsilon: %.6f\n", epsilon)
	fmt.Printf("  alpha: %.6f\n", alpha)
	fmt.Printf("  Input level: %d\n", x.Level())

	fmt.Printf("\nInput values (first 10):\n")
	for i := 0; i < 10 && i < len(values); i++ {
		fmt.Printf("  [%d]: x=%.6f, expected 1/sqrt(x)=%.6f\n", i, values[i], expected[i])
	}

	// Track current en and kn for display
	currentEn := epsilon
	currentKn := 0.0

	// Helper to compute Kn (for display)
	computeKn := func(en float64) float64 {
		a := 1 - en*en*en
		b := 6*en*en - 6
		c := 9 - 9*en
		d := 0.0
		roots := solveCubic(a, b, c, d)
		if len(roots) >= 2 {
			rootIdx := 1
			if rootIdx >= len(roots) {
				rootIdx = len(roots) - 1
			}
			return real(roots[rootIdx])
		}
		return 2.0
	}

	// Create debug callback that decrypts and compares at each iteration
	debugCallback := func(stage string, iteration int, ct *rlwe.Ciphertext, isAn bool) {
		decrypted := tc.decryptCiphertext(t, ct)

		fmt.Printf("\n═══════════════════════════════════════════════════════════════\n")
		fmt.Printf("%s - Iteration %d:\n", stage, iteration)
		fmt.Printf("  Current en: %.10e\n", currentEn)
		if currentKn > 0 {
			fmt.Printf("  Current kn: %.10f\n", currentKn)
		}

		// For an: show value
		// For bn: compare with expected 1/sqrt(x)
		nCompare := 10
		if nCompare > len(values) {
			nCompare = len(values)
		}

		fmt.Printf("  Decrypted values (first %d):\n", nCompare)
		for i := 0; i < nCompare; i++ {
			if isAn {
				fmt.Printf("    [%d]: an=%.10f (x=%.6f)\n", i, decrypted[i], paddedValues[i])
			} else {
				diff := math.Abs(decrypted[i] - expected[i])
				relErr := 0.0
				if expected[i] != 0 {
					relErr = diff / expected[i] * 100
				}
				fmt.Printf("    [%d]: CKKS=%.10f, Plaintext=%.10f, diff=%.2e (%.2f%%)\n",
					i, decrypted[i], expected[i], diff, relErr)
			}
		}

		// Compute statistics for bn
		if !isAn {
			var sumDiff, sumRelErr float64
			maxDiff := 0.0
			nValid := 0
			for i := 0; i < len(values); i++ {
				diff := math.Abs(decrypted[i] - expected[i])
				sumDiff += diff
				if expected[i] != 0 {
					relErr := diff / math.Abs(expected[i])
					sumRelErr += relErr
					nValid++
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			if nValid > 0 {
				avgDiff := sumDiff / float64(len(values))
				avgRelErr := sumRelErr / float64(nValid)
				fmt.Printf("  Statistics: avg_diff=%.2e, avg_rel_err=%.2f%%, max_diff=%.2e\n",
					avgDiff, avgRelErr*100, maxDiff)

				// Calculate precision in bits
				precision := -math.Log2(avgDiff)
				fmt.Printf("  Precision: %.2f bits\n", precision)
			}

			// Update en for next iteration
			if iteration > 0 {
				kn := currentKn
				currentEn = kn * currentEn * math.Pow(3.0-kn*currentEn, 2.0) / 4.0
				currentKn = computeKn(currentEn)
			}
		}
	}

	// Initialize k0
	currentKn = computeKn(currentEn)

	// Run InvSqrt with debug callback
	fmt.Printf("\nStarting InvSqrt computation with debug callback...\n")
	fmt.Printf("Initial k0: %.10f\n", currentKn)
	fmt.Printf("Initial 3/k0: %.10f\n", 3.0/currentKn)
	fmt.Printf("Initial k0^1.5/2: %.10f\n", math.Pow(currentKn, 1.5)/2.0)
	fmt.Printf("Initial k0^3/4: %.10f\n", math.Pow(currentKn, 3.0)/4.0)
	fmt.Printf("\n")

	tc.llama.InvSqrt(x, epsilon, alpha, debugCallback)

	fmt.Printf("\n═══════════════════════════════════════════════════════════════\n")
	fmt.Printf("Debug analysis complete.\n")
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")
}

// TestInvSqrt_Basic tests basic functionality of InvSqrt
func TestInvSqrt_Basic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupTestContext(t, getDefaultTestParams())

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
			name:         "THOR compatible range [0.2, 0.5, 0.8]",
			values:       []float64{0.2, 0.5, 0.8},
			epsilon:      0.15,
			alpha:        0.001,
			level:        8,
			maxMSE:       1e-3,
			minPrecision: 10.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if *debugMode {
				fmt.Printf("\n=== Test: %s ===\n", tt.name)
				fmt.Printf("Parameters: epsilon=%.3f, alpha=%.3f, level=%d\n",
					tt.epsilon, tt.alpha, tt.level)
			}

			// Encrypt input
			x := tc.encryptValues(t, tt.values, tt.level)

			if *debugMode && tt.name == "THOR compatible range [0.2, 0.5, 0.8]" {
				// Run debug version for THOR compatible values
				debugInvSqrt(t, tc, x, tt.values, tt.epsilon, tt.alpha)
			}

			// Compute InvSqrt on ciphertext
			yCt := tc.llama.InvSqrt(x, tt.epsilon, tt.alpha)

			// Compute InvSqrt on plaintext
			// Pad values to match MaxSlots by repeating the input values cyclically
			paddedValues := make([]float64, tc.params.MaxSlots())
			for i := 0; i < tc.params.MaxSlots(); i++ {
				paddedValues[i] = tt.values[i%len(tt.values)]
			}
			yPt := tc.llama.InvSqrtPlaintext(paddedValues)

			// Decrypt and compare
			yCtDecrypted := tc.decryptCiphertext(t, yCt)
			compareWithPlaintext(t, yCtDecrypted, yPt, tt.maxMSE, tt.minPrecision)
		})
	}
}

// TestInvSqrt_ParameterCombinations tests different epsilon and alpha combinations
func TestInvSqrt_ParameterCombinations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupTestContext(t, getDefaultTestParams())

	tests := []struct {
		name         string
		epsilon      float64
		alpha        float64
		maxMSE       float64
		minPrecision float64
	}{
		{
			name:         "THOR paper values (epsilon=0.015, alpha=0.001)",
			epsilon:      0.015,
			alpha:        0.001,
			maxMSE:       1e-3,
			minPrecision: 10.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if *debugMode {
				fmt.Printf("\n=== Test: %s ===\n", tt.name)
			}

			values := generateTestValues(-20, 0, tc.params.MaxSlots())
			x := tc.encryptValues(t, values, 5)

			// Compute InvSqrt
			start := time.Now()
			yCt := tc.llama.InvSqrt(x, tt.epsilon, tt.alpha)
			elapsed := time.Since(start)

			if *debugMode {
				fmt.Printf("Time: %v\n", elapsed)
			}

			// Compute plaintext result
			// Compute plaintext result with cyclic repetition
			paddedValues := make([]float64, tc.params.MaxSlots())
			for i := 0; i < tc.params.MaxSlots(); i++ {
				paddedValues[i] = values[i%len(values)]
			}
			yPt := tc.llama.InvSqrtPlaintext(paddedValues)

			// Compare
			yCtDecrypted := tc.decryptCiphertext(t, yCt)
			compareWithPlaintext(t, yCtDecrypted, yPt, tt.maxMSE, tt.minPrecision)
		})
	}
}

// TestInvSqrt_Performance tests performance characteristics
func TestInvSqrt_Performance(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupTestContext(t, getDefaultTestParams())

	if *debugMode {
		fmt.Printf("\n=== Performance Test ===\n")
	}

	values := generateTestValues(0.15, 10, tc.params.MaxSlots())
	x := tc.encryptValues(t, values, 5)

	// Measure ciphertext computation
	start := time.Now()
	yCt := tc.llama.InvSqrt(x, 0.015, 0.001)
	ctTime := time.Since(start)

	// Measure plaintext computation with cyclic repetition
	paddedValues := make([]float64, tc.params.MaxSlots())
	for i := 0; i < tc.params.MaxSlots(); i++ {
		paddedValues[i] = values[i%len(values)]
	}
	start = time.Now()
	_ = tc.llama.InvSqrtPlaintext(paddedValues)
	ptTime := time.Since(start)

	if *debugMode {
		fmt.Printf("Ciphertext time: %v\n", ctTime)
		fmt.Printf("Plaintext time: %v\n", ptTime)
		fmt.Printf("Slowdown factor: %.2fx\n", float64(ctTime)/float64(ptTime))
		fmt.Printf("Input level: %d, Output level: %d\n", x.Level(), yCt.Level())
	}

	// Ciphertext should complete in reasonable time
	require.Less(t, ctTime.Seconds(), 60.0,
		"InvSqrt took too long: %v", ctTime)
}

// BenchmarkInvSqrt benchmarks the InvSqrt function
func BenchmarkInvSqrt(b *testing.B) {
	tc := setupTestContext(&testing.T{}, getDefaultTestParams())

	values := generateTestValues(-20, 0, tc.params.MaxSlots())
	x := tc.encryptValues(&testing.T{}, values, 5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tc.llama.InvSqrt(x.CopyNew(), 0.1, 0.5)
	}
}

// BenchmarkInvSqrtPlaintext benchmarks the plaintext version
func BenchmarkInvSqrtPlaintext(b *testing.B) {
	tc := setupTestContext(&testing.T{}, getDefaultTestParams())

	values := make([]float64, tc.params.MaxSlots())
	for i := range values {
		values[i] = sampling.RandFloat64(0.1, 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tc.llama.InvSqrtPlaintext(values)
	}
}
