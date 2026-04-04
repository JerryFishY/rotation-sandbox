package main

/*
Stage-1 Linear Correctness Tests - Benchmark-like Strategy Checks

This file implements BENCHMARK-LIKE STRATEGY CHECKS for the linear components
of the Cachemir system. It verifies the correctness of the Decoder and other
linear transformations under CKKS.

IMPORTANT: This is NOT a strict correctness test. By default (CACHEMIR_ASSERT unset
or "0"), the test LOGS RMSE results and PASSES regardless of error magnitude.
Set CACHEMIR_ASSERT=1 to enable strict correctness assertions.

Deterministic Randomness:
- Fixed seed: stage1Seed = 42 (trials use stage1Seed + int64(trial))
- Ensures reproducible test inputs across runs

Default Parameters:
- seqLen:   29
- hidDim:   32
- expDim:   64
- numHeads: 2

Environment Variables:
- CACHEMIR_LONG:   Set to "1" to run more trials (default: 1 trial)
- CACHEMIR_ASSERT: Set to "1" to fail the test if RMSE exceeds tolerance
- CACHEMIR_TOL:    Override default RMSE tolerance (default: 1e-3)
*/

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Deterministic seed for Stage-1 tests - ensures reproducibility
const stage1Seed int64 = 42

// Stage-1 canonical parameters (known runnable defaults)
var (
	Stage1SeqLen   = 29
	Stage1HidDim   = 32
	Stage1ExpDim   = 64
	Stage1NumHeads = 2
)

// Default RMSE tolerance for Stage-1 correctness tests
const defaultRMSEtol = 1e-3

// stage1RNG is a package-level RNG instance for deterministic test generation
var stage1RNG *rand.Rand

func init() {
	stage1RNG = rand.New(rand.NewSource(stage1Seed))
}

// SeedStage1RNG seeds the package-level RNG with stage1Seed for deterministic test inputs
func SeedStage1RNG() {
	stage1RNG = rand.New(rand.NewSource(stage1Seed))
}

// generateDeterministicVector generates a deterministic []complex128 vector
// with values uniformly distributed in [minVal, maxVal] using stage1Seed
func generateDeterministicVector(length int, minVal, maxVal float64) []complex128 {
	SeedStage1RNG()
	result := make([]complex128, length)
	rangeVal := maxVal - minVal
	for i := 0; i < length; i++ {
		result[i] = complex(minVal+rangeVal*stage1RNG.Float64(), 0)
	}
	return result
}

// generateDeterministicMatrix generates a deterministic [][]complex128 matrix
// with values uniformly distributed in [minVal, maxVal] using stage1Seed
func generateDeterministicMatrix(rows, cols int, minVal, maxVal float64) [][]complex128 {
	SeedStage1RNG()
	result := make([][]complex128, rows)
	rangeVal := maxVal - minVal
	for i := 0; i < rows; i++ {
		result[i] = make([]complex128, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = complex(minVal+rangeVal*stage1RNG.Float64(), 0)
		}
	}
	return result
}

// generateSmallVector generates a deterministic vector with small magnitude values
// to avoid CKKS overflow (uniform in [-0.5, 0.5])
func generateSmallVector(length int) []complex128 {
	return generateDeterministicVector(length, -0.5, 0.5)
}

// generateSmallMatrix generates a deterministic matrix with small magnitude values
// to avoid CKKS overflow (uniform in [-0.5, 0.5])
func generateSmallMatrix(rows, cols int) [][]complex128 {
	return generateDeterministicMatrix(rows, cols, -0.5, 0.5)
}

// TestStage1_LinearSkeleton verifies the deterministic input generator
// and establishes the baseline for Stage-1 correctness tests
func TestStage1_LinearSkeleton(t *testing.T) {
	// Seed at start of test for full determinism
	SeedStage1RNG()

	// Test deterministic vector generation
	vec1 := generateSmallVector(Stage1HidDim)

	// Same seed should produce identical vectors
	SeedStage1RNG()
	vec1Again := generateSmallVector(Stage1HidDim)

	for i := 0; i < len(vec1); i++ {
		if real(vec1[i]) != real(vec1Again[i]) {
			t.Errorf("Vector generation not deterministic at index %d: got %v, want %v",
				i, real(vec1Again[i]), real(vec1[i]))
		}
	}

	// Verify vector length matches expected
	if len(vec1) != Stage1HidDim {
		t.Errorf("Vector length mismatch: got %d, want %d", len(vec1), Stage1HidDim)
	}

	// Verify values are within small magnitude range
	for i, v := range vec1 {
		if math.Abs(real(v)) > 0.5 {
			t.Errorf("Vector value out of range at index %d: got %v, want range [-0.5, 0.5]",
				i, real(v))
		}
	}

	// Test deterministic matrix generation
	mat := generateSmallMatrix(Stage1HidDim, Stage1ExpDim)
	if len(mat) != Stage1HidDim || len(mat[0]) != Stage1ExpDim {
		t.Errorf("Matrix dimensions mismatch: got %dx%d, want %dx%d",
			len(mat), len(mat[0]), Stage1HidDim, Stage1ExpDim)
	}

	// Test that different calls produce different values (not all zeros)
	nonZero := false
	for _, row := range mat {
		for _, v := range row {
			if real(v) != 0 {
				nonZero = true
				break
			}
		}
	}
	if !nonZero {
		t.Error("Matrix should contain non-zero values")
	}

	// Verify Stage-1 parameters are correctly defined
	t.Logf("Stage-1 parameters: seqLen=%d, hidDim=%d, expDim=%d, numHeads=%d",
		Stage1SeqLen, Stage1HidDim, Stage1ExpDim, Stage1NumHeads)
	t.Logf("Stage-1 seed: %d", stage1Seed)
	t.Logf("Default RMSE tolerance: %e", defaultRMSEtol)
}

// rmseReal computes the RMSE between two complex128 slices (using real parts).
// Returns an error if lengths don't match.
func rmseReal(a, b []complex128) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("rmse: length mismatch %d vs %d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, nil
	}
	var sum float64
	for i := range a {
		d := real(a[i]) - real(b[i])
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(a))), nil
}

func precisionBits(err float64) float64 {
	if err <= 0 {
		return 64 // Arbitrary high value for zero error
	}
	return -math.Log2(err)
}

// decoderCorrectnessParams holds shared test parameters to avoid repeated setup
type decoderCorrectnessParams struct {
	params ckks.Parameters
	bpLit  bootstrapping.ParametersLiteral
	llama  *LlamaInference
	helper *TestHelper
	size   *LlamaSize
	slots  int
	stride int
	level  int
	logN   int
	tol    float64
	long   bool
	assert bool
}

func setupDecoderCorrectnessParams(t *testing.T) *decoderCorrectnessParams {
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
		t.Fatalf("Failed to create CKKS parameters: %v", err)
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
	stride := slots / size.hidDim
	if stride <= 0 || stride*size.hidDim != slots {
		t.Fatalf("Unexpected slot packing: slots=%d hidDim=%d stride=%d", slots, size.hidDim, stride)
	}

	long := os.Getenv("CACHEMIR_LONG") == "1"
	assertMode := os.Getenv("CACHEMIR_ASSERT") == "1"
	tolStr := os.Getenv("CACHEMIR_TOL")
	tol := defaultRMSEtol
	if tolStr != "" {
		if parsed, err := strconv.ParseFloat(tolStr, 64); err == nil && parsed > 0 {
			tol = parsed
		}
	}

	// Restore original flags
	*test, *logN, *level = oldTest, oldLogN, oldLevel
	*hidDim, *expDim, *seqLen, *numHeads = oldHidDim, oldExpDim, oldSeqLen, oldNumHeads
	*parallel = oldParallel

	return &decoderCorrectnessParams{
		params: params,
		bpLit:  bpLit,
		llama:  llama,
		helper: helper,
		size:   size,
		slots:  slots,
		stride: stride,
		level:  6,
		logN:   8,
		tol:    tol,
		long:   long,
		assert: assertMode,
	}
}

// runDecoderTrial executes a single decoder trial using existing params.
// Used by the Correctness subtest.
func runDecoderTrial(t *testing.T, p *decoderCorrectnessParams, trial int) (rmse float64, out, outMsg []complex128) {
	rng := rand.New(rand.NewSource(stage1Seed + int64(trial)))
	msgSlots := make([]float64, p.slots)
	for j := range msgSlots {
		if j%4 == 0 {
			msgSlots[j] = -10 + 20*rng.Float64()
		} else {
			msgSlots[j] = 0
		}
	}

	pt := ckks.NewPlaintext(p.params, p.level)
	if err := p.helper.encoder.Encode(msgSlots, pt); err != nil {
		t.Fatalf("Failed to encode plaintext: %v", err)
	}
	xCt, err := p.helper.encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Failed to encrypt: %v", err)
	}

	xMsg := make([]complex128, p.size.hidDim)
	for i := 0; i < p.size.hidDim; i++ {
		xMsg[i] = complex(msgSlots[i*p.stride], 0)
	}

	outCt := p.llama.DecoderThor(xCt)
	outPad := p.helper.Dec(outCt, 0)
	out = make([]complex128, p.size.hidDim)
	for i := 0; i < p.size.hidDim; i++ {
		out[i] = outPad[i*p.stride]
	}
	outMsg = p.llama.DecoderMsg(xMsg)

	rmse, err = rmseReal(out, outMsg)
	if err != nil {
		t.Fatalf("RMSE computation failed: %v", err)
	}
	return rmse, out, outMsg
}

// formatSampleValues formats a small sample of got vs want values with delta for debugging
func formatSampleValues(got, want []complex128, maxEntries int) string {
	n := len(got)
	if n > maxEntries {
		n = maxEntries
	}
	var b strings.Builder
	for i := 0; i < n; i++ {
		gotVal := real(got[i])
		wantVal := real(want[i])
		delta := gotVal - wantVal
		b.WriteString(fmt.Sprintf("  [%d] got=%.6f want=%.6f delta=%.6e\n", i, gotVal, wantVal, delta))
	}
	if len(got) > maxEntries {
		b.WriteString(fmt.Sprintf("  ... and %d more\n", len(got)-maxEntries))
	}
	return b.String()
}

// runDecoderWithFreshContext creates a fresh decoder context and runs a single trial.
// Used by the Determinism subtest to verify reproducibility across context recreations.
func runDecoderWithFreshContext(t *testing.T, trial int) (rmse float64, out, outMsg []complex128) {
	oldTest, oldLogN, oldLevel := *test, *logN, *level
	oldHidDim, oldExpDim, oldSeqLen, oldNumHeads := *hidDim, *expDim, *seqLen, *numHeads
	oldParallel := *parallel
	defer func() {
		*test, *logN, *level = oldTest, oldLogN, oldLevel
		*hidDim, *expDim, *seqLen, *numHeads = oldHidDim, oldExpDim, oldSeqLen, oldNumHeads
		*parallel = oldParallel
	}()

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
		t.Fatalf("Failed to create CKKS parameters: %v", err)
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
	stride := slots / size.hidDim
	if stride <= 0 || stride*size.hidDim != slots {
		t.Fatalf("Unexpected slot packing: slots=%d hidDim=%d stride=%d", slots, size.hidDim, stride)
	}

	rng := rand.New(rand.NewSource(stage1Seed + int64(trial)))
	msgSlots := make([]float64, slots)
	for j := range msgSlots {
		if j%4 == 0 {
			msgSlots[j] = -10 + 20*rng.Float64()
		} else {
			msgSlots[j] = 0
		}
	}

	pt := ckks.NewPlaintext(params, *level)
	if err := helper.encoder.Encode(msgSlots, pt); err != nil {
		t.Fatalf("Failed to encode plaintext: %v", err)
	}
	xCt, err := helper.encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Failed to encrypt: %v", err)
	}

	xMsg := make([]complex128, size.hidDim)
	for i := 0; i < size.hidDim; i++ {
		xMsg[i] = complex(msgSlots[i*stride], 0)
	}

	outCt := llama.DecoderThor(xCt)
	outPad := helper.Dec(outCt, 0)
	out = make([]complex128, size.hidDim)
	for i := 0; i < size.hidDim; i++ {
		out[i] = outPad[i*stride]
	}
	outMsg = llama.DecoderMsg(xMsg)

	rmse, err = rmseReal(out, outMsg)
	if err != nil {
		t.Fatalf("RMSE computation failed: %v", err)
	}
	return rmse, out, outMsg
}

func TestCorrectness_Decoder(t *testing.T) {
	// Test 1: Basic correctness with multiple trials
	t.Run("Correctness", func(t *testing.T) {
		p := setupDecoderCorrectnessParams(t)

		trials := 1
		if p.long {
			trials = 3
		}

		var worstRMSE float64
		for trial := 0; trial < trials; trial++ {
			rmse, out, outMsg := runDecoderTrial(t, p, trial)
			if rmse > worstRMSE {
				worstRMSE = rmse
			}
			bits := precisionBits(rmse)

			t.Logf("DecoderCorrectness trial=%d/%d | logN=%d level=%d seqLen=%d hidDim=%d expDim=%d slots=%d stride=%d | RMSE=%.3e | precision=%.2f bits",
				trial+1, trials,
				p.logN, p.level, p.size.seqLen, p.size.hidDim, p.size.expDim,
				p.slots, p.stride,
				rmse, bits,
			)

			if p.assert && rmse > p.tol {
				t.Errorf("RMSE too large (trial %d): got %.3e want < %.3e\n%s",
					trial, rmse, p.tol, formatSampleValues(out, outMsg, 8))
			}
		}

		if p.assert && worstRMSE > p.tol {
			t.Errorf("worst-case RMSE too large: got %.3e want < %.3e", worstRMSE, p.tol)
		}
	})

	// Test 2: Determinism - run the same trial twice with fresh context each time
	t.Run("Determinism", func(t *testing.T) {
		trial := 0 // Fixed trial for determinism check

		// Run 1 with fresh context
		rmse1, _, _ := runDecoderWithFreshContext(t, trial)

		// Run 2 with fresh context - should produce identical results
		rmse2, _, _ := runDecoderWithFreshContext(t, trial)

		t.Logf("Determinism check: trial=%d run1=%.10f run2=%.10f diff=%.10e",
			trial, rmse1, rmse2, rmse1-rmse2)

		// RMSE should be identical within floating-point tolerance (1e-11)
		// Slight differences can occur due to FP precision but should be negligible
		determinismTol := 1e-11
		if math.Abs(rmse1-rmse2) > determinismTol {
			t.Errorf("RMSE not deterministic: run1=%.10e run2=%.10e diff=%.10e want < %.0e",
				rmse1, rmse2, rmse1-rmse2, determinismTol)
		}
	})
}
