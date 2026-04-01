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

	"cachemir_linear/helper"
)

var (
	debugLayerNormMode = flag.Bool("debug-ln", false, "enable debug mode for LayerNorm with verbose output")
)

// testLayerNormParameters holds the test CKKS parameters for LayerNorm
type testLayerNormParameters struct {
	LogN            int
	LogQ            []int
	LogP            []int
	LogDefaultScale int
	Xs              ring.Ternary
}

// getDefaultLayerNormTestParams returns default test parameters for LayerNorm
func getDefaultLayerNormTestParams() testLayerNormParameters {
	return testLayerNormParameters{
		LogN:            12,
		LogQ:            []int{52, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192},
	}
}

// testLayerNormContext holds the testing context for LayerNorm
type testLayerNormContext struct {
	params  *ckks.Parameters
	llama   *LlamaInference
	helper  *TestHelper
	size    *LlamaSize
	btpEval *bootstrapping.Evaluator
}

// setupLayerNormTestContext creates a test context with CKKS parameters for LayerNorm
func setupLayerNormTestContext(t *testing.T, params testLayerNormParameters) *testLayerNormContext {
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

	return &testLayerNormContext{
		params:  &ckksParams,
		llama:   llama,
		helper:  helper,
		size:    size,
		btpEval: llama.btpEval,
	}
}

// generateGammaBeta generates gamma and beta parameters for LayerNorm
func generateGammaBeta(n int) (gamma, beta []float64) {
	gamma = make([]float64, n)
	beta = make([]float64, n)

	for i := 0; i < n; i++ {
		// Gamma typically initialized to 1.0, small variations
		gamma[i] = 1.0 + sampling.RandFloat64(-0.1, 0.1)
		// Beta typically initialized to 0.0
		beta[i] = sampling.RandFloat64(-0.1, 0.1)
	}
	return
}

// encryptValuesForLayerNorm encrypts float64 values into a ciphertext for LayerNorm
// Uses COLUMN-MAJOR layout: message[i + j*hidDim] for j in groups, i in [0, hidDim)
// Input values should be in column-major format (length maxSlots)
func (tc *testLayerNormContext) encryptValuesForLayerNorm(t *testing.T, values []float64, level int) *rlwe.Ciphertext {
	maxSlots := tc.params.MaxSlots()
	message := make([]float64, maxSlots)

	// Use COLUMN-MAJOR layout matching Norm/NormPlaintext
	// For MaxSlots=2048, hidDim=256: numGroups = 2048/256 = 8
	//
	// Data layout (column-major):
	// message[i + j*hidDim] for i in [0, 256), j in [0, 8)
	//
	// This creates:
	// message[0], message[256], message[512], ..., message[1792] = values[0] (same position across groups)
	// message[1], message[257], message[513], ..., message[1793] = values[1]
	// ... (256 positions, 8 groups each)
	//
	// Norm's rotation uses stride = MaxSlots/hidDim = 8:
	//   Rotation by 8 swaps elements across groups:
	//   - position 0 (group0,pos0) <-> position 8 (group1,pos0)
	//   - position 1 (group0,pos1) <-> position 9 (group1,pos1)
	//   After all rotations (8, 16, 32, ..., 1024), all 8 groups' values[i] are summed
	//
	// For mean computation at position i (i < 256):
	//   - After rotation by 8:  sums position 0 and 8 (both are different group values for same position)
	//   - After rotation by 16: sums position 0, 8, 16, 24 (4 groups)
	//   - Eventually: sums all 8 groups' values for position i
	//   - Final result at position i: sum(groups[0..7][i])

	// values should already be in column-major layout
	if len(values) != maxSlots {
		panic(fmt.Sprintf("encryptValuesForLayerNorm: expected %d values, got %d", maxSlots, len(values)))
	}

	// Copy values directly (already in column-major format)
	copy(message, values)

	plaintext := ckks.NewPlaintext(*tc.params, level)
	err := tc.helper.encoder.Encode(message, plaintext)
	require.NoError(t, err, "failed to encode values")

	ciphertext, err := tc.helper.encryptor.EncryptNew(plaintext)
	require.NoError(t, err, "failed to encrypt values")

	return ciphertext
}

// decryptCiphertextForLayerNorm decrypts a ciphertext to float64 values for LayerNorm
func (tc *testLayerNormContext) decryptCiphertextForLayerNorm(t *testing.T, ct *rlwe.Ciphertext) []float64 {
	msg := make([]complex128, ct.Slots())
	err := tc.helper.encoder.Decode(tc.helper.decryptor.DecryptNew(ct), msg)
	require.NoError(t, err, "failed to decrypt ciphertext")

	result := make([]float64, len(msg))
	for i, v := range msg {
		result[i] = real(v)
	}
	return result
}

// encryptGammaBeta encrypts gamma and beta into ciphertexts
// Uses COLUMN-MAJOR layout: message[i + j*hidDim] for j in groups, i in [0, hidDim)
// Input gamma/beta should be in column-major format (length maxSlots)
func (tc *testLayerNormContext) encryptGammaBeta(t *testing.T, gamma, beta []float64, level int) (*rlwe.Ciphertext, *rlwe.Ciphertext) {
	maxSlots := tc.params.MaxSlots()

	// Validate input lengths
	if len(gamma) != maxSlots {
		panic(fmt.Sprintf("encryptGammaBeta: expected gamma length %d, got %d", maxSlots, len(gamma)))
	}
	if len(beta) != maxSlots {
		panic(fmt.Sprintf("encryptGammaBeta: expected beta length %d, got %d", maxSlots, len(beta)))
	}

	// Use COLUMN-MAJOR layout (input should already be in this format)
	// Copy directly without transformation
	gammaPadded := make([]float64, maxSlots)
	copy(gammaPadded, gamma)

	betaPadded := make([]float64, maxSlots)
	copy(betaPadded, beta)

	// Encrypt gamma
	ptGamma := ckks.NewPlaintext(*tc.params, level)
	err := tc.helper.encoder.Encode(gammaPadded, ptGamma)
	require.NoError(t, err, "failed to encode gamma")
	ctGamma, err := tc.helper.encryptor.EncryptNew(ptGamma)
	require.NoError(t, err, "failed to encrypt gamma")

	// Encrypt beta
	ptBeta := ckks.NewPlaintext(*tc.params, level)
	err = tc.helper.encoder.Encode(betaPadded, ptBeta)
	require.NoError(t, err, "failed to encode beta")
	ctBeta, err := tc.helper.encryptor.EncryptNew(ptBeta)
	require.NoError(t, err, "failed to encrypt beta")

	return ctGamma, ctBeta
}

// computeNormThorPlaintext computes the plaintext reference for NormThor
// This matches the ciphertext computation including THOR pre-scaling and numerically stable variance
//
// Data layout: values[position*numGroups + group] = position i, group j
//
//	indices 0-7: position 0, groups 0-7
//	indices 8-15: position 1, groups 0-7
//	...
//
// Rotation aggregates across positions within each group (same as Norm/NormPlaintext)
func computeNormThorPlaintext(xValues, gammaValues, betaValues []float64, hidDim int, config LayerNormConfig) []float64 {
	n := len(xValues)
	numGroups := n / hidDim
	result := make([]float64, n)
	nFloat := float64(hidDim)

	// THOR pre-scaling parameters
	wBuffer := 1.05
	maxForDenominator := (config.MaxVar*wBuffer + config.VarEpsilon) * math.Pow(float64(hidDim), 2)
	scaleFactor := 1.0 / math.Sqrt(maxForDenominator)
	scaledEps := config.VarEpsilon / maxForDenominator

	// Step 1: Apply THOR pre-scaling
	xScaled := make([]float64, n)
	for i := 0; i < n; i++ {
		xScaled[i] = xValues[i] * scaleFactor
	}

	// Step 2: Compute sumX (sum across positions for each group)
	// Data layout: xScaled[position*numGroups + group]
	// For group g: sum = xScaled[g] + xScaled[g+numGroups] + xScaled[g+2*numGroups] + ...
	sumX := make([]float64, n)
	for g := 0; g < numGroups; g++ {
		groupSum := 0.0
		// Sum across all positions for this group
		for p := 0; p < hidDim; p++ {
			groupSum += xScaled[p*numGroups+g]
		}
		// After aggregation, all positions in this group get the same sum
		for p := 0; p < hidDim; p++ {
			sumX[p*numGroups+g] = groupSum
		}
	}

	// Step 3: Compute sumXSq (sum of squared values across positions for each group)
	sumXSq := make([]float64, n)
	for g := 0; g < numGroups; g++ {
		groupSum := 0.0
		// Sum squares across all positions for this group
		for p := 0; p < hidDim; p++ {
			val := xScaled[p*numGroups+g]
			groupSum += val * val
		}
		// After aggregation, all positions in this group get the same sum
		for p := 0; p < hidDim; p++ {
			sumXSq[p*numGroups+g] = groupSum
		}
	}

	// Step 4: Compute variance using numerically stable formula
	// variance = N * sumXSq - (sumX)^2 + scaledEps
	variance := make([]float64, n)
	for i := 0; i < n; i++ {
		termA := nFloat * sumXSq[i]
		termB := sumX[i] * sumX[i]
		variance[i] = termA - termB + scaledEps
	}

	// Step 5: Compute result = (N * x_scaled - sumX) / sqrt(variance) * gamma + beta
	for i := 0; i < n; i++ {
		numerator := nFloat*xScaled[i] - sumX[i]
		normalized := numerator / math.Sqrt(variance[i])
		result[i] = gammaValues[i]*normalized + betaValues[i]
	}

	return result
}

// compareLayerNormWithPlaintext compares ciphertext result with plaintext result for LayerNorm
func compareLayerNormWithPlaintext(t *testing.T, ctResult, ptResult []float64, maxMSE float64, minPrecisionBits float64) {
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

	// Check MSE
	require.Less(t, mse, maxMSE,
		"MSE %f exceeds threshold %f", mse, maxMSE)

	// Calculate precision in bits
	precision := -math.Log2(mse)
	require.Greater(t, precision, minPrecisionBits,
		"precision %.2f bits is less than required %.2f bits", precision, minPrecisionBits)

	if *debugLayerNormMode {
		fmt.Printf("  ✓ MSE: %e, Precision: %.2f bits\n", mse, precision)
		fmt.Printf("  ✓ Max difference: %e\n", maxDiff)

		// Show first 10 values side-by-side
		fmt.Printf("  ✓ Sample comparison (first 10 values):\n")
		fmt.Printf("    Index | Expected (CKKS) | Actual (Plaintext) | Difference | Relative Error\n")
		fmt.Printf("    -------|----------------|-------------------|------------|----------------\n")
		for i := 0; i < 10 && i < len(ctResult); i++ {
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

// debugLayerNorm runs LayerNorm with detailed debugging of each stage
func debugLayerNorm(t *testing.T, tc *testLayerNormContext, x *rlwe.Ciphertext, gamma, beta *rlwe.Ciphertext,
	xValues, gammaValues, betaValues []float64, config LayerNormConfig) {

	fmt.Printf("\n╔════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║      DEBUG MODE: LayerNorm Step-by-Step Comparison            ║\n")
	fmt.Printf("╚════════════════════════════════════════════════════════════════╝\n\n")

	fmt.Printf("Configuration:\n")
	fmt.Printf("  Type: %d\n", config.Type)
	fmt.Printf("  MinVar: %.6f, MaxVar: %.6f\n", config.MinVar, config.MaxVar)
	fmt.Printf("  N: %d\n", config.N)
	fmt.Printf("  Epsilon: %.10f\n", config.VarEpsilon)

	epsilonVar := config.MinVar / config.MaxVar
	wBuffer := 1.05
	maxForDenominator := (config.MaxVar*wBuffer + config.VarEpsilon) * math.Pow(float64(config.N), 2)
	scaleFactor := 1.0 / math.Sqrt(maxForDenominator)

	fmt.Printf("  EpsilonVar: %.10f\n", epsilonVar)
	fmt.Printf("  ScaleFactor: %.10f\n", scaleFactor)
	fmt.Printf("  MaxForDenominator: %.6f\n", maxForDenominator)

	// Prepare plaintext values (full column-major layout for debugging)
	// xValues, gammaValues, betaValues already contain maxSlots values in column-major format
	maxSlots := tc.params.MaxSlots()
	n := config.N

	// Extract first N values for xPt (these are the values for group 0)
	xPt := make([]float64, n)
	gammaPt := make([]float64, n)
	betaPt := make([]float64, n)
	for i := 0; i < n; i++ {
		xPt[i] = xValues[i]
		gammaPt[i] = gammaValues[i]
		betaPt[i] = betaValues[i]
	}

	// Helper to compare ciphertext with plaintext at each step
	compareStep := func(stepName string, ct *rlwe.Ciphertext, expectedPt []float64) {
		fmt.Printf("\n═══════════════════════════════════════════════════════════════\n")
		fmt.Printf("Step: %s (level=%d)\n", stepName, ct.Level())

		decrypted := tc.decryptCiphertextForLayerNorm(t, ct)
		nCompare := 10
		if nCompare > n {
			nCompare = n
		}

		fmt.Printf("  Comparison (first %d values):\n", nCompare)
		fmt.Printf("    Index | CKKS          | Plaintext     | Difference    | Rel Error\n")
		fmt.Printf("    -------|---------------|---------------|---------------|----------\n")

		var sumDiff, sumRelErr float64
		nValid := 0

		for i := 0; i < nCompare; i++ {
			diff := decrypted[i] - expectedPt[i]
			relErr := 0.0
			if expectedPt[i] != 0 {
				relErr = math.Abs(diff/expectedPt[i]) * 100
				nValid++
				sumRelErr += relErr
			}
			sumDiff += math.Abs(diff)

			fmt.Printf("    %5d | %13.6f | %13.6f | %13.2e | %8.2f%%\n",
				i, decrypted[i], expectedPt[i], diff, relErr)
		}

		// Calculate statistics for all N values
		var totalDiff, totalRelErr float64
		totalValid := 0
		for i := 0; i < n; i++ {
			diff := math.Abs(decrypted[i] - expectedPt[i])
			totalDiff += diff
			if expectedPt[i] != 0 {
				relErr := diff / math.Abs(expectedPt[i]) * 100
				totalRelErr += relErr
				totalValid++
			}
		}

		avgDiff := totalDiff / float64(n)
		avgRelErr := 0.0
		if totalValid > 0 {
			avgRelErr = totalRelErr / float64(totalValid)
		}
		mse := math.Sqrt(totalDiff*totalDiff) / float64(n)

		fmt.Printf("  Statistics: avg_diff=%.2e, avg_rel_err=%.2f%%, mse=%.2e\n",
			avgDiff, avgRelErr, mse)
	}

	// Compute expected plaintext values at each step
	// IMPORTANT: The ciphertext uses the numerically-stable formula:
	//   variance = N * sum(x^2) - (sum(x))^2
	//   numerator = N * x_scaled - sumX
	//   result = (numerator / sqrt(variance)) * gamma + beta
	//
	// Data layout: values[position*numGroups + group] = position i, group j
	//   indices 0-7: position 0, groups 0-7
	//   indices 8-15: position 1, groups 0-7
	//   ...
	//
	// Rotation (by batchSize=8) swaps elements within the same group:
	//   rotation by 8: position 0 <-> position 1 (same group)
	//   After all rotations: sum of all positions for each group

	hidDim := tc.size.hidDim
	numGroups := maxSlots / hidDim // Number of groups = 2048/256 = 8
	nFloat := float64(hidDim)

	// Debug: print actual maxSlots
	fmt.Printf("  [DEBUG] maxSlots: %d\n", maxSlots)
	fmt.Printf("  [DEBUG] hidDim: %d, numGroups: %d\n", hidDim, numGroups)
	fmt.Printf("  [DEBUG] len(xValues): %d, len(gammaValues): %d, len(betaValues): %d\n",
		len(xValues), len(gammaValues), len(betaValues))
	fmt.Printf("  [DEBUG] First 5 xValues: %.6f, %.6f, %.6f, %.6f, %.6f\n",
		xValues[0], xValues[1], xValues[2], xValues[3], xValues[4])

	// Step 1: After THOR pre-scaling
	// x_scaled = x * scaleFactor
	xScaledPt := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		xScaledPt[i] = xValues[i] * scaleFactor
	}

	// Step 2: After computing sumX (sum across positions for each group)
	// Data layout: values[position*numGroups + group]
	// Rotation aggregates across positions within each group
	// For group g: sum = xScaledPt[g] + xScaledPt[g+8] + xScaledPt[g+16] + ... + xScaledPt[g+2040]
	sumXPt := make([]float64, maxSlots)
	for g := 0; g < numGroups; g++ {
		groupSum := 0.0
		// Sum across all positions for this group
		for p := 0; p < hidDim; p++ {
			groupSum += xScaledPt[p*numGroups+g]
		}
		// After aggregation, all positions in this group get the same sum
		for p := 0; p < hidDim; p++ {
			sumXPt[p*numGroups+g] = groupSum
		}
	}

	// Step 3: After computing nx = N * x_scaled
	nxPt := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		nxPt[i] = nFloat * xScaledPt[i]
	}

	// Step 4: After computing numerator = nx - sumX
	numeratorPt := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		numeratorPt[i] = nxPt[i] - sumXPt[i]
	}

	// Step 5: After computing variance = N * sum(x^2) - (sum(x))^2
	// First compute sumXSq (sum of squared values across positions for each group)
	sumXSqPt := make([]float64, maxSlots)
	for g := 0; g < numGroups; g++ {
		groupSum := 0.0
		// Sum squares across all positions for this group
		for p := 0; p < hidDim; p++ {
			val := xScaledPt[p*numGroups+g]
			groupSum += val * val
		}
		// After aggregation, all positions in this group get the same sum
		for p := 0; p < hidDim; p++ {
			sumXSqPt[p*numGroups+g] = groupSum
		}
	}

	// Then compute variance using the numerically stable formula
	// variance = N * sumXSq - (sumX)^2
	// Note: scaledEps is added in the ciphertext but not in the plaintext for comparison
	variancePt := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		termA := nFloat * sumXSqPt[i]
		termB := sumXPt[i] * sumXPt[i]
		variancePt[i] = termA - termB
	}

	// Step 6: Final result = (numerator / sqrt(variance)) * gamma + beta
	finalPt := make([]float64, maxSlots)
	for i := 0; i < maxSlots; i++ {
		normalized := numeratorPt[i] / math.Sqrt(variancePt[i])
		finalPt[i] = gammaValues[i]*normalized + betaValues[i]
	}

	// Now create debug callback to capture intermediate ciphertexts
	var xScaledCt, sumXCt, numeratorCt, varianceCt *rlwe.Ciphertext

	debugCallback := func(stage string, data interface{}) {
		switch stage {
		case "After THOR pre-scaling":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				xScaledCt = ct.CopyNew()
				compareStep(stage, xScaledCt, xScaledPt)
			}
		case "After computing sumX":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				sumXCt = ct.CopyNew()
				compareStep(stage, sumXCt, sumXPt)
			}
		case "After computing nx = N * x_scaled":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				compareStep(stage, ct, nxPt)
			}
		case "After computing numerator":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				numeratorCt = ct.CopyNew()
				compareStep(stage, numeratorCt, numeratorPt)
			}
		case "After computing variance":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				varianceCt = ct.CopyNew()
				compareStep(stage, varianceCt, variancePt)
			}
		case "After x * invSqrt":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				// Compute expected plaintext for numerator/sqrt(variance)
				// Note: ciphertext uses numerator (N*x - sumX), not just x
				numeratorOverSqrtVarPt := make([]float64, maxSlots)
				for i := 0; i < maxSlots; i++ {
					numeratorOverSqrtVarPt[i] = numeratorPt[i] / math.Sqrt(variancePt[i])
				}
				compareStep(stage, ct, numeratorOverSqrtVarPt)
			}
		case "After multiplying by gamma":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				// Compute expected plaintext for (numerator/sqrt(variance))*gamma
				scaledPt := make([]float64, maxSlots)
				for i := 0; i < maxSlots; i++ {
					normalized := numeratorPt[i] / math.Sqrt(variancePt[i])
					scaledPt[i] = gammaValues[i] * normalized
				}
				compareStep(stage, ct, scaledPt)
			}
		case "After adding beta":
			if ct, ok := data.(*rlwe.Ciphertext); ok {
				// This is the final result
				compareStep(stage, ct, finalPt)
			}
		default:
			// Handle InvSqrt stages gracefully
			if len(stage) > 8 && stage[:8] == "InvSqrt:" {
				// For InvSqrt intermediate stages, skip comparison
				_ = data
			}
		}
	}

	fmt.Printf("\nStarting NormThor computation with step-by-step comparison...\n")
	result := tc.llama.NormThor(x, 0, debugCallback)

	// Compare final result
	compareStep("Final result", result, finalPt)

	fmt.Printf("\n═══════════════════════════════════════════════════════════════\n")
	fmt.Printf("Debug analysis complete.\n")
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")
}

// TestLayerNorm_Basic tests basic functionality of LayerNorm
func TestLayerNorm_Basic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupLayerNormTestContext(t, getDefaultLayerNormTestParams())

	tests := []struct {
		name         string
		config       LayerNormConfig
		minVar       float64
		maxVar       float64
		level        int
		maxMSE       float64
		minPrecision float64
	}{
		{
			name:         "LayerNorm1: min_var=0.15, max_var=10",
			config:       GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm1),
			minVar:       0.15,
			maxVar:       10.0,
			level:        10,
			maxMSE:       1e-2,
			minPrecision: 8.0,
		},
		{
			name:         "LayerNorm2: min_var=0.2, max_var=150",
			config:       GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm2),
			minVar:       0.2,
			maxVar:       150.0,
			level:        10,
			maxMSE:       1e-2,
			minPrecision: 8.0,
		},
		{
			name:         "LayerNorm3: min_var=0.75, max_var=2500",
			config:       GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm3),
			minVar:       0.75,
			maxVar:       2500.0,
			level:        10,
			maxMSE:       1e-2,
			minPrecision: 8.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if *debugLayerNormMode {
				fmt.Printf("\n=== Test: %s ===\n", tt.name)
				fmt.Printf("Config: Type=%d, minVar=%.6f, maxVar=%.6f, N=%d\n",
					tt.config.Type, tt.config.MinVar, tt.config.MaxVar, tt.config.N)
			}

			// Generate test values with target variance range
			n := tt.config.N
			maxSlots := tc.params.MaxSlots()
			numGroups := maxSlots / n

			// Generate test values in column-major layout with proper variance
			xValuesFull := helper.GenerateTestValuesWithVarianceColumnMajor(tt.minVar, tt.maxVar, n, numGroups)

			// Generate gamma/beta values (constant across positions)
			gammaValues, betaValues := generateGammaBeta(n)

			// Expand gamma/beta to full message size in column-major layout
			gammaValuesFull := make([]float64, maxSlots)
			betaValuesFull := make([]float64, maxSlots)
			for i := 0; i < n; i++ {
				for j := 0; j < numGroups; j++ {
					gammaValuesFull[i+j*n] = gammaValues[i]
					betaValuesFull[i+j*n] = betaValues[i]
				}
			}

			// Encrypt inputs (data is already in column-major format)
			x := tc.encryptValuesForLayerNorm(t, xValuesFull, tt.level)
			gamma, beta := tc.encryptGammaBeta(t, gammaValuesFull, betaValuesFull, tt.level)

			if *debugLayerNormMode {
				// Run debug version with full column-major arrays
				debugLayerNorm(t, tc, x, gamma, beta, xValuesFull, gammaValuesFull, betaValuesFull, tt.config)
			}

			// Compute NormThor on ciphertext (THOR pre-scaling is applied inside NormThor)
			yCt := tc.llama.NormThor(x, 0)

			// Decrypt result
			yCtDecrypted := tc.decryptCiphertextForLayerNorm(t, yCt)

			// Compute expected plaintext result (with THOR scaling)
			yPt := computeNormThorPlaintext(xValuesFull, gammaValuesFull, betaValuesFull, n, tt.config)

			// Compare with plaintext
			if *debugLayerNormMode {
				fmt.Printf("\n✓ NormThor computation completed\n")
				fmt.Printf("  Input level: %d, Output level: %d\n", x.Level(), yCt.Level())
			}

			compareLayerNormWithPlaintext(t, yCtDecrypted, yPt, tt.maxMSE, tt.minPrecision)
		})
	}
}

// TestLayerNorm_AllThreeRanges tests all three THOR input ranges
func TestLayerNorm_AllThreeRanges(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupLayerNormTestContext(t, getDefaultLayerNormTestParams())

	// Test all three configurations from THOR
	configs := []struct {
		name         string
		config       LayerNormConfig
		minVar       float64
		maxVar       float64
		maxMSE       float64
		minPrecision float64
	}{
		{
			name:         "THOR ln1: [0.15, 10]",
			config:       GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm1),
			minVar:       0.15,
			maxVar:       10.0,
			maxMSE:       1e-2,
			minPrecision: 8.0,
		},
		{
			name:         "THOR ln2: [0.2, 150]",
			config:       GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm2),
			minVar:       0.2,
			maxVar:       150.0,
			maxMSE:       1e-2,
			minPrecision: 8.0,
		},
		{
			name:         "THOR ln3: [0.75, 2500]",
			config:       GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm3),
			minVar:       0.75,
			maxVar:       2500.0,
			maxMSE:       1e-2,
			minPrecision: 8.0,
		},
	}

	for _, tt := range configs {
		t.Run(tt.name, func(t *testing.T) {
			if *debugLayerNormMode {
				fmt.Printf("\n=== Testing %s ===\n", tt.name)
			}

			n := tt.config.N
			maxSlots := tc.params.MaxSlots()
			numGroups := maxSlots / n

			// Generate test values in column-major layout with proper variance
			xValuesFull := helper.GenerateTestValuesWithVarianceColumnMajor(tt.minVar, tt.maxVar, n, numGroups)

			// Generate gamma/beta values (constant across positions)
			gammaValues, betaValues := generateGammaBeta(n)

			// Expand gamma/beta to full message size in column-major layout
			gammaValuesFull := make([]float64, maxSlots)
			betaValuesFull := make([]float64, maxSlots)
			for i := 0; i < n; i++ {
				for j := 0; j < numGroups; j++ {
					gammaValuesFull[i+j*n] = gammaValues[i]
					betaValuesFull[i+j*n] = betaValues[i]
				}
			}

			// Encrypt inputs (data is already in column-major format)
			x := tc.encryptValuesForLayerNorm(t, xValuesFull, 10)
			// gamma, beta := tc.encryptGammaBeta(t, gammaValuesFull, betaValuesFull, 10)

			// Compute NormThor
			start := time.Now()
			yCt := tc.llama.NormThor(x, 0)
			elapsed := time.Since(start)

			// Decrypt and compare
			yCtDecrypted := tc.decryptCiphertextForLayerNorm(t, yCt)
			yPt := computeNormThorPlaintext(xValuesFull, gammaValuesFull, betaValuesFull, n, tt.config)

			if *debugLayerNormMode {
				fmt.Printf("Time: %v\n", elapsed)
			}

			// Verify correctness
			compareLayerNormWithPlaintext(t, yCtDecrypted, yPt, tt.maxMSE, tt.minPrecision)

			// Performance check
			require.Less(t, elapsed.Seconds(), 120.0,
				"NormThor took too long: %v", elapsed)
		})
	}
}

// TestLayerNorm_Performance tests performance characteristics
func TestLayerNorm_Performance(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	tc := setupLayerNormTestContext(t, getDefaultLayerNormTestParams())

	if *debugLayerNormMode {
		fmt.Printf("\n=== NormThor Performance Test ===\n")
	}

	config := GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm1)
	n := config.N
	maxSlots := tc.params.MaxSlots()
	numGroups := maxSlots / n

	// Generate test values in column-major layout with proper variance
	xValuesFull := helper.GenerateTestValuesWithVarianceColumnMajor(config.MinVar, config.MaxVar, n, numGroups)

	// Generate gamma/beta values (constant across positions)
	gammaValues, betaValues := generateGammaBeta(n)

	// Expand gamma/beta to full message size in column-major layout
	gammaValuesFull := make([]float64, maxSlots)
	betaValuesFull := make([]float64, maxSlots)
	for i := 0; i < n; i++ {
		for j := 0; j < numGroups; j++ {
			gammaValuesFull[i+j*n] = gammaValues[i]
			betaValuesFull[i+j*n] = betaValues[i]
		}
	}

	// Encrypt inputs (data is already in column-major format)
	x := tc.encryptValuesForLayerNorm(t, xValuesFull, 10)
	// gamma, beta := tc.encryptGammaBeta(t, gammaValuesFull, betaValuesFull, 10)

	// Measure ciphertext computation
	start := time.Now()
	yCt := tc.llama.NormThor(x, 0)
	ctTime := time.Since(start)

	// Measure plaintext computation
	start = time.Now()
	_ = computeNormThorPlaintext(xValuesFull, gammaValuesFull, betaValuesFull, n, config)
	ptTime := time.Since(start)

	if *debugLayerNormMode {
		fmt.Printf("Ciphertext time: %v\n", ctTime)
		fmt.Printf("Plaintext time: %v\n", ptTime)
		fmt.Printf("Slowdown factor: %.2fx\n", float64(ctTime)/float64(ptTime))
		fmt.Printf("Input level: %d, Output level: %d\n", x.Level(), yCt.Level())
	}

	// Ciphertext should complete in reasonable time
	require.Less(t, ctTime.Seconds(), 120.0,
		"NormThor took too long: %v", ctTime)
}

// BenchmarkLayerNorm benchmarks the NormThor function
func BenchmarkLayerNorm(b *testing.B) {
	tc := setupLayerNormTestContext(&testing.T{}, getDefaultLayerNormTestParams())

	config := GetDefaultLayerNormConfigForLlama(tc.llama, LayerNorm1)
	n := config.N
	maxSlots := tc.params.MaxSlots()
	numGroups := maxSlots / n

	// Generate test values in column-major layout with proper variance
	xValuesFull := helper.GenerateTestValuesWithVarianceColumnMajor(config.MinVar, config.MaxVar, n, numGroups)

	// Generate gamma/beta values (constant across positions)
	gammaValues, betaValues := generateGammaBeta(n)

	// Expand gamma/beta to full message size in column-major layout
	gammaValuesFull := make([]float64, maxSlots)
	betaValuesFull := make([]float64, maxSlots)
	for i := 0; i < n; i++ {
		for j := 0; j < numGroups; j++ {
			gammaValuesFull[i+j*n] = gammaValues[i]
			betaValuesFull[i+j*n] = betaValues[i]
		}
	}

	// Encrypt inputs (data is already in column-major format)
	x := tc.encryptValuesForLayerNorm(&testing.T{}, xValuesFull, 10)
	// gamma, beta := tc.encryptGammaBeta(&testing.T{}, gammaValuesFull, betaValuesFull, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tc.llama.NormThor(x.CopyNew(), 0)
	}
}
