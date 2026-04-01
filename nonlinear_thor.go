package main

import (
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Package-level cache for THOR exponential polynomial approximations
var (
	exp8xSmallOnce  sync.Once
	exp8xSmallPoly  polynomial.PolynomialVector
	exp8xSmallEval  *polynomial.Evaluator
	exp8xSmallScale *big.Float
	exp8xSmallConst *big.Float

	exp8xLargeOnce  sync.Once
	exp8xLargePoly  polynomial.PolynomialVector
	exp8xLargeEval  *polynomial.Evaluator
	exp8xLargeScale *big.Float
	exp8xLargeConst *big.Float
)

// THOR HE_EXP1 coefficients (Low-to-High order, degree 15)
// Range: [-27.2493, 21.72692], Input Scaling: 1/32
var thorExp1Coeffs = []float64{
	0.0006522770224130905, // c0 = constant term
	0.005218196900295354,
	0.020873931555133732,
	0.05566510463488879,
	0.11128698173597897,
	0.1780344447042791,
	0.2379982262906916,
	0.27220647178765545,
	0.26787311936472025,
	0.23721029007596767,
	0.20618261949210986,
	0.15202099984697098,
	0.0670090353368128,
	0.03881607331549499,
	0.05948672763856172,
	0.032855468333339584, // c15
}

// THOR HE_EXP2 coefficients (Low-to-High order, degree 15)
// Range: [-70, 70], Input Scaling: 1/64

var thorExp2Coeffs = []float64{
	8.615994668877663e-05, // c0 = constant term
	0.0006877176070101981,
	0.0027930565779849003,
	0.00749909544008294,
	0.014216636671807894,
	0.022268203231858744,
	0.03517495704921328,
	0.04217318306400685,
	0.02336452059478656,
	0.016604320800445653,
	0.04817928878801878,
	0.0397324053296174,
	-0.009262268572236316,
	-0.008386712802267769,
	0.014226972463907047,
	0.008201736399899691, // c15
}

// cExp8xSmall computes c0 * exp(8*x) approximation for x in [-27.2493, 21.72692]
// where c0 = 0.0006522770224130905 (THOR's constant term coefficient)
// This uses THOR's HE_EXP1 monomial polynomial approximation (degree 15)
// Input should be pre-scaled by 1/32 (handled by upstream softmax)
//
// The function caches the polynomial vector and evaluator on first call.
// By default, processes all slots when no mask is provided.
//
// Parameters:
//
//	x - Input ciphertext (pre-scaled by 1/32)
//	iterations - Number of squaring iterations (y = y^2^iterations). Default 0 (no squaring).
//
// Returns:
//
//	y - Ciphertext containing c0 * exp(8*x) approximation raised to power 2^iterations
func (llama *LlamaInference) cExp8xSmall(x *rlwe.Ciphertext, iterations int) (y *rlwe.Ciphertext) {
	var err error
	Debug := true
	eval := llama.eval[0]
	fmt.Printf("Computing cExp8xSmall (THOR HE_EXP1, iterations=%d)...\n", iterations)
	start := time.Now()

	// Lazy initialization using sync.Once
	exp8xSmallOnce.Do(func() {
		fmt.Print("Preparing cExp8xSmall polynomial (THOR HE_EXP1, Monomial basis)...\n")

		// Convert THOR's coefficients to []*big.Float
		prec := llama.params.EncodingPrecision()
		coeffsBig := make([]*big.Float, len(thorExp1Coeffs))
		for i, c := range thorExp1Coeffs {
			coeffsBig[i] = bignum.NewFloat(c, prec)
		}

		// Create Monomial polynomial directly (no interval needed)
		monoPoly := bignum.NewPolynomial(bignum.Monomial, coeffsBig, nil)
		// monoPoly.IsOdd = false
		// monoPoly.IsEven = false

		// Create targetMapping with full slots (default case for no mask)
		fullSlots := make([]int, llama.params.MaxSlots())
		for i := range fullSlots {
			fullSlots[i] = i
		}
		targetMapping := map[int][]int{0: fullSlots}

		// Create and cache polynomial vector
		polyVec, err := polynomial.NewPolynomialVector(
			[]bignum.Polynomial{monoPoly},
			targetMapping,
		)
		if err != nil {
			panic(err)
		}

		// Store in package-level cache
		exp8xSmallPoly = polyVec

		// Create and cache evaluator
		exp8xSmallEval = polynomial.NewEvaluator(*llama.params, eval)

		// Compute and cache scale/constant
		scale, constant := polyVec.ChangeOfBasis(llama.params.MaxSlots())
		exp8xSmallScale = scale[0]
		exp8xSmallConst = constant[0]

		if Debug {
			fmt.Printf("  [DEBUG cExp8xSmall] ChangeOfBasis scale: %.15e (expected 1.0)\n", scale[0])
			fmt.Printf("  [DEBUG cExp8xSmall] ChangeOfBasis constant: %.15e (expected 0.0)\n", constant[0])
		}
	})

	// Step 1: Apply input transformation: x' = x - mid_x (NO division - pre-scaled by upstream)
	// midX must also be pre-scaled since the input x is pre-scaled
	midXOriginal := (-27.2493 + 21.72692) / 2.0 // = -2.76119 (original midpoint)
	midX := midXOriginal / 32.0                 // Pre-scaled midpoint
	negMidX := -midX
	eval.Add(x, negMidX, x)

	eval.Mul(x, exp8xSmallScale, x)
	if !IsIntegerBig(exp8xSmallScale) {
		eval.Rescale(x, x)
	}
	if exp8xSmallConst.Sign() != 0 {
		eval.Add(x, exp8xSmallConst, x)
	}

	y, err = exp8xSmallEval.Evaluate(x, exp8xSmallPoly, llama.params.DefaultScale())
	if err != nil {
		panic(err)
	}

	// Step 3: Square y = y^2 for specified iterations
	// Each iteration computes y = y * y (squaring doubles the exponent)
	for i := 0; i < iterations; i++ {
		eval.MulRelin(y, y, y)
		eval.Rescale(y, y)
	}

	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n",
		elapsed.Seconds(), x.Level(), y.Level())

	return y
}

// cExp8xLarge computes c0 * exp(8*x) approximation for x in [-64.0, 64.0]
// where c0 = 8.615994668877663e-05 (THOR's constant term coefficient)
// This uses THOR's HE_EXP2 hardcoded monomial coefficients (degree 15)
// Input should be pre-scaled by 1/64 (handled by upstream softmax)
//
// Parameters:
//
//	x - Input ciphertext (pre-scaled by 1/64)
//	iterations - Number of squaring iterations (y = y^2^iterations). Default 0 (no squaring).
//
// Returns:
//
//	y - Ciphertext containing c0 * exp(8*x) approximation raised to power 2^iterations
func (llama *LlamaInference) cExp8xLarge(x *rlwe.Ciphertext, iterations int) (y *rlwe.Ciphertext) {
	Debug := true
	eval := llama.eval[0]
	fmt.Printf("Computing cExp8xLarge (THOR HE_EXP2, Monomial basis, iterations=%d)...\n", iterations)
	start := time.Now()

	// Lazy initialization using sync.Once
	exp8xLargeOnce.Do(func() {
		fmt.Print("Preparing cExp8xLarge polynomial (THOR HE_EXP2, Monomial basis)...\n")

		// Convert THOR's coefficients to []*big.Float
		prec := llama.params.EncodingPrecision()
		coeffsBig := make([]*big.Float, len(thorExp2Coeffs))
		for i, c := range thorExp2Coeffs {
			coeffsBig[i] = bignum.NewFloat(c, prec)
		}

		// Create Monomial polynomial directly (no interval needed)
		monoPoly := bignum.NewPolynomial(bignum.Monomial, coeffsBig, nil)

		// Debug output
		if Debug {
			fmt.Printf("  [DEBUG cExp8xLarge] Using THOR HE_EXP2 coefficients\n")
			fmt.Printf("  [DEBUG cExp8xLarge] Basis: Monomial (no Chebyshev approximation)\n")
			fmt.Printf("  [DEBUG cExp8xLarge] Degree: %d\n", monoPoly.Degree())
			fmt.Printf("  [DEBUG cExp8xLarge] Constant term c0: %.15e\n", thorExp2Coeffs[len(thorExp2Coeffs)-1])
			fmt.Printf("  [DEBUG cExp8xLarge] Input range: [%.6f, %.6f]\n", -70.0, 70.0)
			fmt.Printf("  [DEBUG cExp8xLarge] Pre-scale factor: 1/64\n")
		}

		// Setup polynomial vector
		nonZero := make([]int, llama.params.MaxSlots())
		for i := 0; i < llama.params.MaxSlots(); i++ {
			nonZero[i] = i
		}
		mapping := map[int][]int{0: nonZero}

		poly, _ := polynomial.NewPolynomialVector(
			[]bignum.Polynomial{monoPoly},
			mapping,
		)

		polyeval := polynomial.NewEvaluator(*llama.params, llama.eval[0])
		scale, constant := poly.ChangeOfBasis(llama.params.MaxSlots())

		// For Monomial basis: scale should be 1.0, constant should be 0.0
		if Debug {
			fmt.Printf("  [DEBUG cExp8xLarge] ChangeOfBasis scale: %.15e (expected 1.0)\n", scale[0])
			fmt.Printf("  [DEBUG cExp8xLarge] ChangeOfBasis constant: %.15e (expected 0.0)\n", constant[0])
		}

		exp8xLargePoly = poly
		exp8xLargeEval = polyeval
		exp8xLargeScale = scale[0]
		exp8xLargeConst = constant[0]
	})

	// Step 1: Apply input transformation: x' = x - mid_x
	// midX = (-70.0 + 70.0) / 2.0 = 0, so we skip subtracting 0
	// (no centering needed for symmetric range)

	// Step 2: Apply ChangeOfBasis (no-op for Monomial: scale=1, constant=0)
	// Keep for consistency, even though scale=1, constant=0
	eval.Mul(x, exp8xLargeScale, x)
	if !IsIntegerBig(exp8xLargeScale) {
		eval.Rescale(x, x)
	}
	if exp8xLargeConst.Sign() != 0 {
		eval.Add(x, exp8xLargeConst, x)
	}

	// Step 3: Evaluate polynomial
	var err error
	y, err = exp8xLargeEval.Evaluate(x, exp8xLargePoly, llama.params.DefaultScale())
	if err != nil {
		panic(err)
	}

	// Step 4: Square y = y^2 for specified iterations
	// Each iteration computes y = y * y (squaring doubles the exponent)
	for i := 0; i < iterations; i++ {
		eval.MulRelin(y, y, y)
		eval.Rescale(y, y)
	}

	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n",
		elapsed.Seconds(), x.Level(), y.Level())

	return y
}

func (llama *LlamaInference) cExp8xSmallPlaintext(x []complex128) []float64 {
	// Computes the true function c0 * exp(8*x) for comparison with polynomial approximation
	// This is the target function that the polynomial approximates
	c0 := thorExp1Coeffs[0]
	midXOriginal := (-27.2493 + 21.72692) / 2.0
	midX := midXOriginal / 32.0 // Pre-scaled midpoint
	result := make([]float64, len(x))
	for i, val := range x {
		xFloat := real(val)
		xPoly := xFloat - midX             // Center the input (same transformation as ciphertext version)
		result[i] = c0 * math.Exp(8*xPoly) // True function value
	}
	return result
}

func (llama *LlamaInference) cExp8xLargePlaintext(x []complex128) []float64 {
	c0 := thorExp2Coeffs[0]
	// midX = (-70.0 + 70.0) / 2.0 = 0, so no centering needed
	result := make([]float64, len(x))
	for i, val := range x {
		xFloat := real(val)
		// NO division here (pre-scaled by test)
		result[i] = c0 * math.Exp(8*xFloat)
	}
	return result
}

// DebugInvSqrtCallback is called during InvSqrt iterations when debug mode is enabled
// stage: description of the current computation stage
// iteration: current iteration number
// ct: ciphertext at this stage (can be an or bn)
// isAn: true if ct is an, false if ct is bn
type DebugInvSqrtCallback func(stage string, iteration int, ct *rlwe.Ciphertext, isAn bool)

func (llama *LlamaInference) InvSqrt(x *rlwe.Ciphertext, epsilon float64, alpha float64, debugCB ...DebugInvSqrtCallback) *rlwe.Ciphertext {
	eval := llama.eval[0]
	fmt.Print("Computing InvSqrt with CKKS-optimized algorithm...\n")
	start := time.Now()

	// Extract debug callback if provided
	var debug DebugInvSqrtCallback
	if len(debugCB) > 0 && debugCB[0] != nil {
		debug = debugCB[0]
	}

	en := epsilon
	iteration := 0
	maxIterations := 10
	const minMulDepthForIteration = 2 // Minimum levels needed for one iteration

	// Helper function to compute relaxation factor by solving cubic equation
	// Equation: (1 - en^3) * k^3 + (6*en^2 - 6) * k^2 + (9 - 9*en) * k = 0
	// This is the THOR algorithm's robust implementation
	computeKn := func(en float64) float64 {
		// Coefficients for: (1 - en^3)k^3 + (6en^2 - 6)k^2 + (9 - 9en)k = 0
		a := 1 - en*en*en
		b := 6*en*en - 6
		c := 9 - 9*en
		d := 0.0

		// Debug: Print coefficients for first call
		if debug != nil && en == epsilon {
			fmt.Printf("  [DEBUG computeKn] en=%.6f\n", en)
			fmt.Printf("  [DEBUG computeKn] Coefficients: a=%.6f, b=%.6f, c=%.6f, d=%.6f\n", a, b, c, d)
		}

		// Solve the cubic equation
		roots := solveCubic(a, b, c, d)

		// Debug: Print all roots
		if debug != nil && en == epsilon {
			fmt.Printf("  [DEBUG computeKn] All roots:\n")
			for i, root := range roots {
				fmt.Printf("    root[%d]: %.10f + %.10fi\n", i, real(root), imag(root))
			}
		}

		// Filter for real, positive roots > 1 (since we need to expand the range [en, 1])
		var validRoots []float64
		for _, root := range roots {
			if math.Abs(imag(root)) < 1e-10 { // Real root
				realPart := real(root)
				if realPart > 1.0+1e-6 {
					validRoots = append(validRoots, realPart)
				}
			}
		}

		// Debug: Print valid roots
		if debug != nil && en == epsilon {
			fmt.Printf("  [DEBUG computeKn] Valid roots (>1): %v\n", validRoots)
		}

		// Fallback: if no valid roots, return 3.0 (for e -> 0 or 1 cases)
		if len(validRoots) == 0 {
			if debug != nil && en == epsilon {
				fmt.Printf("  [DEBUG computeKn] No valid roots, returning fallback 3.0\n")
			}
			return 3.0
		}

		// Heuristic: Pick the smaller root for stability.
		// The smaller root corresponds to k approx 2 (when en=0.15), barrier ~1.5.
		// This provides a higher barrier (3/k) preventing 'an' from crossing the singularity.
		minK := validRoots[0]
		for _, k := range validRoots {
			if k < minK {
				minK = k
			}
		}

		if debug != nil && en == epsilon {
			fmt.Printf("  [DEBUG computeKn] Selected k0: %.10f\n", minK)
		}

		// Return the minimum valid root (smaller k = larger barrier = more stable)
		return minK
	}

	// Initialize a0 = x (an starts as x)
	an := x.CopyNew()

	// b0 = 1.0 (plaintext, not ciphertext)
	// We'll skip the first iteration's bn computation since b0=1 simplifies the first update
	// b1 = 1 * (k0^1.5/2) * (3/k0 - a0)

	// First iteration (unrolled from loop to avoid computing bn as ciphertext)
	k0 := computeKn(en)

	// Debug callback for initial state
	if debug != nil {
		debug("an (initial)", 0, an, true)
	}

	// Compute: term = 3/k0 - a0
	// Since Sub(ciphertext, scalar) = ciphertext - scalar, we compute:
	// 3/k0 - a0 = -(a0 - 3/k0)
	threeOverK0 := 3.0 / k0
	termA0, err := eval.SubNew(an, threeOverK0) // a0 - 3/k0
	if err != nil {
		panic(fmt.Errorf("failed to compute (a0 - 3/k0): %v", err))
	}
	// Negate to get 3/k0 - a0 (this does NOT change the scale)
	llama.helper.Neg(eval, termA0) // termA0 = 3/k0 - a0

	// Debug callback for termA0
	if debug != nil {
		debug("termA0 = 3/k0 - a0", 0, termA0, false)
	}

	// b1 = 1 * (k0^1.5/2) * termA0 = (k0^1.5/2) * termA0
	k015Val := math.Pow(k0, 1.5) / 2.0
	var bn *rlwe.Ciphertext
	bn, err = eval.MulNew(termA0, k015Val) // bn = b1
	if err != nil {
		panic(fmt.Errorf("failed to compute b1: %v", err))
	}
	eval.Rescale(bn, bn)

	// Debug callback for b1
	if debug != nil {
		debug("b1 = (k0^1.5/2) * termA0", 1, bn, false)
	}

	// a1 = a0 * (k0^3/4) * termA0^2
	// Compute termA0^2
	an2, err := eval.MulRelinNew(termA0, termA0)
	if err != nil {
		panic(fmt.Errorf("failed to compute termA0^2: %v", err))
	}
	eval.Rescale(an2, an2)

	// Compute a0 * (k0^3/4)
	k03Val := math.Pow(k0, 3.0) / 4.0
	an1, err := eval.MulNew(an, k03Val)
	if err != nil {
		panic(fmt.Errorf("failed to compute a0 * (k0^3/4): %v", err))
	}
	eval.Rescale(an1, an1)

	// Ensure levels match before final multiplication
	if an1.Level() > an2.Level() {
		eval.DropLevel(an1, an1.Level()-an2.Level())
	} else if an2.Level() > an1.Level() {
		eval.DropLevel(an2, an2.Level()-an1.Level())
	}

	// a1 = an1 * an2
	an, err = eval.MulRelinNew(an1, an2)
	if err != nil {
		panic(fmt.Errorf("failed to compute a1: %v", err))
	}
	eval.Rescale(an, an)

	// Debug callback for a1
	if debug != nil {
		debug("a1 = a0 * (k0^3/4) * termA0^2", 1, an, true)
	}

	// Update error bound
	en = k0 * en * math.Pow(3.0-k0*en, 2.0) / 4.0
	iteration++

	// Main iteration loop
	for en < (1.0-alpha) && iteration < maxIterations {
		// Check if we need bootstrap (require at least 2 levels for multiplication depth)
		if an.Level() < minMulDepthForIteration || bn.Level() < minMulDepthForIteration {
			fmt.Printf("Not enough levels (an=%d, bn=%d), bootstrapping...\n", an.Level(), bn.Level())
			an = llama.BootTo(an, llama.params.MaxLevel())
			bn = llama.BootTo(bn, llama.params.MaxLevel())
		}

		iteration++
		kn := computeKn(en)

		// Compute: term = 3/kn - an
		threeOverKn := 3.0 / kn
		term, err := eval.SubNew(an, threeOverKn) // an - 3/kn
		if err != nil {
			panic(fmt.Errorf("failed to compute (an - 3/kn): %v", err))
		}
		// Negate to get 3/kn - an (this does NOT change the scale)
		llama.helper.Neg(eval, term) // term = 3/kn - an

		// Compute term^2 (needed for a_{n+1})
		termSq, err := eval.MulRelinNew(term, term)
		if err != nil {
			panic(fmt.Errorf("failed to compute term^2: %v", err))
		}
		eval.Rescale(termSq, termSq)

		// Update b: b_{n+1} = bn * (kn^1.5/2) * term
		// Step 1: bn * (kn^1.5/2)
		kn15Val := math.Pow(kn, 1.5) / 2.0
		bn1, err := eval.MulNew(bn, kn15Val)
		if err != nil {
			panic(fmt.Errorf("failed to compute bn * (kn^1.5/2): %v", err))
		}
		eval.Rescale(bn1, bn1)

		// Step 2: multiply by term
		bn, err = eval.MulRelinNew(bn1, term)
		if err != nil {
			panic(fmt.Errorf("failed to compute bn_new: %v", err))
		}
		eval.Rescale(bn, bn)

		// Debug callback for bn
		if debug != nil {
			debug(fmt.Sprintf("b%d", iteration), iteration, bn, false)
		}

		// Update a: a_{n+1} = an * (kn^3/4) * term^2
		// Step 1: an * (kn^3/4)
		kn3Val := math.Pow(kn, 3.0) / 4.0
		an1, err := eval.MulNew(an, kn3Val)
		if err != nil {
			panic(fmt.Errorf("failed to compute an * (kn^3/4): %v", err))
		}
		eval.Rescale(an1, an1)

		// Step 2: multiply by term^2
		an, err = eval.MulRelinNew(an1, termSq)
		if err != nil {
			panic(fmt.Errorf("failed to compute an_new: %v", err))
		}
		eval.Rescale(an, an)

		// Debug callback for an
		if debug != nil {
			debug(fmt.Sprintf("a%d", iteration), iteration, an, true)
		}

		// Update error bound
		en = kn * en * math.Pow(3.0-kn*en, 2.0) / 4.0
	}

	// Debug callback for final result
	if debug != nil {
		debug("Final bn", iteration, bn, false)
	}

	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with %d iterations, input level %d and output level %d\n",
		elapsed.Seconds(), iteration, x.Level(), bn.Level())
	return bn
}

func (llama *LlamaInference) InvSqrtPlaintext(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, val := range x {
		result[i] = 1.0 / math.Sqrt(val)
	}
	return result
}

// DebugInvCallback is called during Inv iterations when debug mode is enabled
// stage: description of the current computation stage
// iteration: current iteration number
// ct: ciphertext at this stage (can be an or bn)
// isAn: true if ct is an, false if ct is bn
type DebugInvCallback func(stage string, iteration int, ct *rlwe.Ciphertext, isAn bool)

// InvConfig holds configuration options for the inverse algorithm
type InvConfig struct {
	Epsilon            float64          // Initial error bound (default: 2^(-11) for softmax)
	Alpha              float64          // Convergence threshold (default: 0.001)
	ConjugateDenoising bool             // Apply conjugate denoising (default: false)
	ComplexPacking     bool             // Pack an and bn for single bootstrap (default: false)
	SignalEnhancement  bool             // Pre-multiply before bootstrap (default: false)
	DebugCallback      DebugInvCallback // Debug callback function
}

// Inv computes 1/x using Goldschmidt's algorithm with adaptive SOR (aSOR).
// The algorithm uses dynamic relaxation factors kn = 2/(en+1) computed from
// the error bound en, which starts at epsilon and converges to 1-alpha.
// Reference: THOR he_inv function from the open-source code.
func (llama *LlamaInference) Inv(x *rlwe.Ciphertext, config InvConfig) *rlwe.Ciphertext {
	eval := llama.eval[0]
	fmt.Print("Computing Inv with CKKS-optimized algorithm...\n")
	start := time.Now()

	// Set defaults
	if config.Epsilon == 0 {
		config.Epsilon = math.Pow(2, -11) // THOR default for softmax
	}
	if config.Alpha == 0 {
		config.Alpha = 0.001
	}

	en := config.Epsilon
	iteration := 0
	maxIterations := 15
	const minMulDepthForIteration = 2 // Minimum levels needed for one iteration

	// Helper function to compute kn = 2/(en+1)
	computeKn := func(en float64) float64 {
		return 2.0 / (en + 1.0)
	}

	// Initialize: bn = x (denominator)
	bn := x.CopyNew()

	// First iteration (special case: an = 1, so we compute a1 directly)
	k0 := computeKn(en)

	// Debug callback for initial state
	if config.DebugCallback != nil {
		config.DebugCallback("bn (initial)", 0, bn, false)
	}

	// Compute: term = 2/k0 - bn
	twoOverK0 := 2.0 / k0
	term, err := eval.SubNew(bn, twoOverK0) // bn - 2/k0
	if err != nil {
		panic(fmt.Errorf("failed to compute (bn - 2/k0): %v", err))
	}
	llama.helper.Neg(eval, term) // term = 2/k0 - bn, \ell

	// Compute a1 = 1 * term * k0^2 (since an = 1)
	k02Val := k0 * k0
	var an *rlwe.Ciphertext
	an, err = eval.MulNew(term, k02Val)
	if err != nil {
		panic(fmt.Errorf("failed to compute a1: %v", err))
	}
	eval.Rescale(an, an) // \ell-1

	// Update bn: b1 = bn * term * k0^2
	bn, err = eval.MulNew(bn, k02Val)
	if err != nil {
		panic(fmt.Errorf("failed to scale b1: %v", err))
	}
	eval.Rescale(bn, bn) // \ell-1

	bn, err = eval.MulRelinNew(bn, term)
	if err != nil {
		panic(fmt.Errorf("failed to compute b1: %v", err))
	}
	eval.Rescale(bn, bn) // \ell-2

	// Debug callback
	if config.DebugCallback != nil {
		config.DebugCallback("a1", 1, an, true)
		config.DebugCallback("b1", 1, bn, false)
	}

	// Update error bound
	en = k0 * en * (2.0 - k0*en)
	iteration++

	// Main iteration loop
	for en < (1.0-config.Alpha) && iteration < maxIterations {
		// Check if we need bootstrap (require at least 2 levels for multiplication depth)
		if an.Level() < minMulDepthForIteration || bn.Level() < minMulDepthForIteration {
			fmt.Printf("Not enough levels (an=%d, bn=%d), bootstrapping...\n", an.Level(), bn.Level())

			// Optimization: Complex packing if enabled
			if config.ComplexPacking {
				// TODO: Implement complex packing for single bootstrap
				// For now, fall back to individual bootstraps
				an = llama.BootTo(an, llama.params.MaxLevel())
				bn = llama.BootTo(bn, llama.params.MaxLevel())
			} else {
				an = llama.BootTo(an, llama.params.MaxLevel())
				bn = llama.BootTo(bn, llama.params.MaxLevel())
			}
		}

		iteration++
		kn := computeKn(en)

		// Compute: term = 2/kn - bn
		twoOverKn := 2.0 / kn
		term, err = eval.SubNew(bn, twoOverKn) // bn - 2/kn
		if err != nil {
			panic(fmt.Errorf("failed to compute (bn - 2/kn): %v", err))
		}
		llama.helper.Neg(eval, term) // term = 2/kn - bn, \ell

		// Optimization: Conjugate denoising
		if config.ConjugateDenoising {
			// Compute conjugate
			conj, err := eval.ConjugateNew(term)
			if err != nil {
				panic(fmt.Errorf("conjugate failed: %v", err))
			}

			// Add: term + conj(term) = 2*Re(term)
			term, err = eval.AddNew(term, conj)
			if err != nil {
				panic(fmt.Errorf("add failed: %v", err))
			}

			// Divide by 2 to get Re(term)
			half := 0.5
			term, err = eval.MulNew(term, half)
			if err != nil {
				panic(fmt.Errorf("scaling failed: %v", err))
			}
			eval.Rescale(term, term)
		}

		// Update an: an = an * term * kn^2
		kn2Val := kn * kn

		// Step 1: an * kn^2
		an1, err := eval.MulNew(an, kn2Val)
		if err != nil {
			panic(fmt.Errorf("failed to compute an * kn^2: %v", err))
		}
		eval.Rescale(an1, an1) // \ell-1

		// Step 2: multiply by term
		an, err = eval.MulRelinNew(an1, term)
		if err != nil {
			panic(fmt.Errorf("failed to compute an_new: %v", err))
		}
		eval.Rescale(an, an) // \ell-2

		// Update bn: bn = bn * term * kn^2
		bn1, err := eval.MulNew(bn, kn2Val)
		if err != nil {
			panic(fmt.Errorf("failed to compute bn * kn^2: %v", err))
		}
		eval.Rescale(bn1, bn1) // \ell-1

		bn, err = eval.MulRelinNew(bn1, term)
		if err != nil {
			panic(fmt.Errorf("failed to compute bn_new: %v", err))
		}
		eval.Rescale(bn, bn) // \ell-2

		// Debug callbacks
		if config.DebugCallback != nil {
			config.DebugCallback(fmt.Sprintf("a%d", iteration), iteration, an, true)
			config.DebugCallback(fmt.Sprintf("b%d", iteration), iteration, bn, false)
		}

		// Update error bound: en = kn * en * (2 - kn*en)
		en = kn * en * (2.0 - kn*en)
	}

	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with %d iterations, input level %d and output level %d\n",
		elapsed.Seconds(), iteration, x.Level(), an.Level())

	return an // an approximates 1/x
}

// InvPlaintext computes 1/x for plaintext values
func (llama *LlamaInference) InvPlaintext(x []float64) []float64 {
	result := make([]float64, len(x))
	for i, val := range x {
		result[i] = 1.0 / val
	}
	return result
}

// LayerNormType defines the three LayerNorm configurations from THOR
type LayerNormType int

const (
	LayerNorm1 LayerNormType = iota // min_var=0.15, max_var=10
	LayerNorm2                      // min_var=0.2, max_var=150
	LayerNorm3                      // min_var=0.75, max_var=2500
)

// LayerNormConfig holds configuration for LayerNorm
type LayerNormConfig struct {
	Type       LayerNormType
	MinVar     float64
	MaxVar     float64
	VarEpsilon float64
	N          int // usually 768 or hidden dimension
}

// GetDefaultLayerNormConfig returns the default config for each LayerNorm type
// Note: These configs were designed for Bert-base (hidDim=768). For Llama (hidDim=256),
// use GetDefaultLayerNormConfigForLlama instead.
func GetDefaultLayerNormConfig(lnType LayerNormType) LayerNormConfig {
	switch lnType {
	case LayerNorm1:
		return LayerNormConfig{
			Type:       LayerNorm1,
			MinVar:     0.15,
			MaxVar:     10.0,
			VarEpsilon: 1e-5,
			N:          768,
		}
	case LayerNorm2:
		return LayerNormConfig{
			Type:       LayerNorm2,
			MinVar:     0.2,
			MaxVar:     150.0,
			VarEpsilon: 1e-5,
			N:          768,
		}
	case LayerNorm3:
		return LayerNormConfig{
			Type:       LayerNorm3,
			MinVar:     0.75,
			MaxVar:     2500.0,
			VarEpsilon: 1e-5,
			N:          768,
		}
	default:
		return GetDefaultLayerNormConfig(LayerNorm1)
	}
}

// GetDefaultLayerNormConfigForLlama returns configs optimized for Llama (hidDim=256)
func GetDefaultLayerNormConfigForLlama(llama *LlamaInference, lnType LayerNormType) LayerNormConfig {
	hidDim := llama.size.hidDim
	switch lnType {
	case LayerNorm1:
		return LayerNormConfig{
			Type:       LayerNorm1,
			MinVar:     0.15,
			MaxVar:     10.0,
			VarEpsilon: 1e-5,
			N:          hidDim, // Use actual hidDim from LlamaSize
		}
	case LayerNorm2:
		return LayerNormConfig{
			Type:       LayerNorm2,
			MinVar:     0.2,
			MaxVar:     150.0,
			VarEpsilon: 1e-5,
			N:          hidDim,
		}
	case LayerNorm3:
		return LayerNormConfig{
			Type:       LayerNorm3,
			MinVar:     0.75,
			MaxVar:     2500.0,
			VarEpsilon: 1e-5,
			N:          hidDim,
		}
	default:
		return GetDefaultLayerNormConfigForLlama(llama, LayerNorm1)
	}
}

// DebugLayerNormCallback is called during LayerNorm computation when debug mode is enabled
type DebugLayerNormCallback func(stage string, data interface{})

func (llama *LlamaInference) NormThor(x *rlwe.Ciphertext, lnType int, debugCB ...DebugLayerNormCallback) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	fmt.Print("Computing NormThor (InvSqrt-based)...\n")
	if llama.normFunc == nil {
		normConfig := GetDefaultLayerNormConfigForLlama(llama, LayerNormType(lnType))
		llama.normFunc = &NormFunc{
			config: normConfig,
			gamma: llama.helper.constCtGen(1.0, x.Level()),
			beta:  llama.helper.constCtGen(0.0, x.Level()),
		}
	}
	config := llama.normFunc.config
	gamma := llama.normFunc.gamma
	beta := llama.normFunc.beta

	start := time.Now()

	// Extract debug callback
	var debug DebugLayerNormCallback
	if len(debugCB) > 0 && debugCB[0] != nil {
		debug = debugCB[0]
	}

	params := llama.params
	hidDim := llama.size.hidDim             // N
	batchSize := params.MaxSlots() / hidDim // B

	// Calculate epsilon for InvSqrt
	epsilonVar := config.MinVar / config.MaxVar

	if debug != nil {
		fmt.Printf("  [DEBUG NormThor] Type: %d\n", config.Type)
		fmt.Printf("  [DEBUG NormThor] min_var: %.6f, max_var: %.6f\n", config.MinVar, config.MaxVar)
		fmt.Printf("  [DEBUG NormThor] epsilon_var: %.10f\n", epsilonVar)
	}

	// ========== THOR Pre-scaling: Normalize input to ensure variance is in (epsilon_var, 1] ==========
	// Calculate max_for_denominator = (max_var * w_buffer + var_e) * n^2
	wBuffer := 1.05
	maxForDenominator := (config.MaxVar*wBuffer + config.VarEpsilon) * math.Pow(float64(config.N), 2)
	scaleFactor := 1.0 / math.Sqrt(maxForDenominator)

	if debug != nil {
		fmt.Printf("  [DEBUG NormThor] THOR scaling factor: %.10e\n", scaleFactor)
		fmt.Printf("  [DEBUG NormThor] max_for_denominator: %.10e\n", maxForDenominator)
	}

	// ========================================================================
	// Step 2: 预缩放输入 (Pre-scaling)
	// ========================================================================
	// Apply THOR pre-scaling to input x
	xScaled, err := eval.MulNew(x, scaleFactor)
	if err != nil {
		panic(fmt.Errorf("failed to apply THOR pre-scaling: %v", err))
	}
	eval.Rescale(xScaled, xScaled)

	if debug != nil {
		debug("After THOR pre-scaling", xScaled)
	}

	// ========== Step 2: Compute 统计量 ==========
	sumX := xScaled.CopyNew()
	for i := batchSize; i < params.MaxSlots(); i *= 2 {
		tmp := sumX.CopyNew()
		// eval.Rotate(tmp, i, tmp)
		rotateAnyStep(eval, tmp, i, tmp)
		eval.Add(tmp, sumX, sumX)
	}

	if debug != nil {
		debug("After computing sumX", sumX)
	}

	xSquared, err := eval.MulRelinNew(xScaled, xScaled) // x^2
	if err != nil {
		panic(err)
	}
	eval.Rescale(xSquared, xSquared)

	sumXSq := xSquared.CopyNew()
	for i := batchSize; i < params.MaxSlots(); i *= 2 {
		tmp := sumXSq.CopyNew()
		// eval.Rotate(tmp, i, tmp)
		rotateAnyStep(eval, tmp, i, tmp)
		eval.Add(tmp, sumXSq, sumXSq)
	}

	// ========================================================================
	// Step 4: 计算 Numerator 和 Variance
	// ========================================================================

	nFloat := float64(hidDim)

	// 4.1 Numerator = N * x_scaled - Sum(x)
	nx, _ := eval.MulNew(xScaled, nFloat)
	// Only rescale if nFloat is not an integer (integer constants don't increase scale in Lattigo)
	if !IsInteger(nFloat) {
		eval.Rescale(nx, nx)
	}

	if debug != nil {
		debug("After computing nx = N * x_scaled", nx)
	}

	numerator, _ := eval.SubNew(nx, sumX)

	if debug != nil {
		debug("After computing numerator", numerator)
	}

	// 4.2 Variance (Scaled) = N * Sum(x^2) - (Sum(x))^2

	// Term A: N * Sum(x^2)
	termA, _ := eval.MulNew(sumXSq, nFloat)
	// Only rescale if nFloat is not an integer
	if !IsInteger(nFloat) {
		eval.Rescale(termA, termA)
	}

	// Term B: (Sum(x))^2
	termB, _ := eval.MulRelinNew(sumX, sumX)
	eval.Rescale(termB, termB)

	variance, _ := eval.SubNew(termA, termB)

	if debug != nil {
		debug("After computing variance", variance)
	}

	// 4.3 加上稳定性因子 (Add Epsilon)
	scaledEps := config.VarEpsilon / maxForDenominator
	eval.Add(variance, scaledEps, variance)

	// ========== Step 5: Compute InvSqrt of variance using THOR's algorithm ==========
	invSqrt := llama.InvSqrt(variance, epsilonVar, 0.001, func(stage string, iteration int, ct *rlwe.Ciphertext, isAn bool) {
		if debug != nil {
			debug(fmt.Sprintf("InvSqrt: %s (iter=%d)", stage, iteration), ct)
		}
	})

	// ========== Step 6: Compute result = x * invSqrt * gamma + beta ==========
	// Note: Use original x, not varc (same as Norm)

	// 6.1 Mul InvSqrt
	res, _ := eval.MulRelinNew(numerator, invSqrt)
	eval.Rescale(res, res)
	if debug != nil {
		debug("After x * invSqrt", res)
	}

	// 6.2 Mul Gamma
	eval.MulRelin(res, gamma, res)
	eval.Rescale(res, res)
	if debug != nil {
		debug("After multiplying by gamma", res)
	}

	// 6.3 Add Beta
	if beta.Level() > res.Level() {
		eval.DropLevel(beta, beta.Level()-res.Level())
	}
	eval.Add(res, beta, res)
	if debug != nil {
		debug("After adding beta", res)
	}

	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n",
		elapsed.Seconds(), x.Level(), res.Level())

	return res
}
