package main

import (
	"fmt"
	"math"
	"math/big"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// solveCubic returns all roots of a cubic equation ax^3 + bx^2 + cx + d = 0
// Using Cardano's formula for cubic equation solving
func solveCubic(a, b, c, d float64) []complex128 {
	// Convert to depressed cubic: t^3 + pt + q = 0
	// where x = t - b/(3a)
	if math.Abs(a) < 1e-10 {
		// If a is close to 0, solve as quadratic
		if math.Abs(b) < 1e-10 {
			// Linear equation: cx + d = 0
			if math.Abs(c) < 1e-10 {
				return []complex128{}
			}
			return []complex128{complex(-d/c, 0)}
		}
		// Quadratic: bx^2 + cx + d = 0
		discriminant := c*c - 4*b*d
		if discriminant < 0 {
			sqrtD := complex(0, math.Sqrt(-discriminant))
			cB := complex(b, 0)
			cC := complex(c, 0)
			return []complex128{
				(-cC + sqrtD) / (2 * cB),
				(-cC - sqrtD) / (2 * cB),
			}
		}
		sqrtD := math.Sqrt(discriminant)
		return []complex128{
			complex((-c+sqrtD)/(2*b), 0),
			complex((-c-sqrtD)/(2*b), 0),
		}
	}

	// Cubic equation coefficients
	p := (3*a*c - b*b) / (3 * a * a)
	q := (2*b*b*b - 9*a*b*c + 27*a*a*d) / (27 * a * a * a)

	discriminant := q*q/4 + p*p*p/27

	var roots []complex128

	if discriminant > 1e-10 {
		// One real root, two complex conjugate roots
		u := math.Cbrt(-q/2 + math.Sqrt(discriminant))
		v := math.Cbrt(-q/2 - math.Sqrt(discriminant))
		t1 := u + v
		x1 := t1 - b/(3*a)

		// Complex roots
		re := -(u + v) / 2
		im := (u - v) * math.Sqrt(3) / 2
		x2 := complex(re-b/(3*a), im)
		x3 := complex(re-b/(3*a), -im)

		roots = []complex128{complex(x1, 0), x2, x3}
	} else if math.Abs(discriminant) <= 1e-10 {
		// Three real roots, at least two equal
		if math.Abs(q) < 1e-10 {
			// Triple root
			x1 := -b / (3 * a)
			roots = []complex128{complex(x1, 0), complex(x1, 0), complex(x1, 0)}
		} else {
			u := math.Cbrt(-q / 2)
			t1 := 2 * u
			t2 := -u
			x1 := t1 - b/(3*a)
			x2 := t2 - b/(3*a)
			roots = []complex128{complex(x1, 0), complex(x2, 0), complex(x2, 0)}
		}
	} else {
		// Three distinct real roots
		r := math.Sqrt(-p * p * p / 27)
		theta := math.Acos(-q / (2 * r))
		t1 := 2 * math.Cbrt(r) * math.Cos(theta/3)
		t2 := 2 * math.Cbrt(r) * math.Cos((theta+2*math.Pi)/3)
		t3 := 2 * math.Cbrt(r) * math.Cos((theta+4*math.Pi)/3)

		x1 := t1 - b/(3*a)
		x2 := t2 - b/(3*a)
		x3 := t3 - b/(3*a)

		roots = []complex128{complex(x1, 0), complex(x2, 0), complex(x3, 0)}
	}

	return roots
}

func rotateAnyStep(eval *ckks.Evaluator, ct *rlwe.Ciphertext, steps int, out *rlwe.Ciphertext) {
	for i := 1; i < ct.Slots(); i *= 2 {
		if steps&i != 0 {
			eval.Rotate(ct, i, out)
			ct = out
		}
	}
}

func PrepareContext(params ckks.Parameters, btpParametersLit bootstrapping.ParametersLiteral) (llama *LlamaInference, helper *TestHelper, size *LlamaSize, opeval *OperationEvaluator) {
	fmt.Print("Preparing context...\n")

	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	num_eval := 1
	if *parallel {
		num_eval = runtime.GOMAXPROCS(0)
	}
	eval := make([]*ckks.Evaluator, num_eval)
	galEls := []uint64{params.GaloisElementForComplexConjugation()}
	for i := 1; i < params.MaxSlots(); i *= 2 {
		galEls = append(galEls, params.GaloisElement(i))
	}
	// Generate Galois keys once and share across all evaluators (keys are read-only)
	sharedEvk := rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...)
	for i := range eval {
		eval[i] = ckks.NewEvaluator(params, sharedEvk)
	}

	var btpEval *bootstrapping.Evaluator
	btpParams, _ := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if *test == "Softmax" || *test == "Norm" || *test == "Model" || *test == "Decoder" || *test == "NormThor" {
		btpEvk, _, _ := btpParams.GenEvaluationKeys(sk)
		btpEval, _ = bootstrapping.NewEvaluator(btpParams, btpEvk)
	}

	helper = &TestHelper{encoder: encoder, encryptor: encryptor, decryptor: decryptor, params: &params}
	size = &LlamaSize{hidDim: *hidDim, expDim: *expDim, numHeads: *numHeads, seqLen: *seqLen}
	llama = &LlamaInference{
		size:    size,
		eval:    eval,
		btpEval: btpEval,
		params:  &params,
		helper:  helper,
		w:       make(map[string][]*rlwe.Plaintext),
		cache:   make(map[string][]*rlwe.Ciphertext),
		mask:    make(map[string][]*rlwe.Plaintext),
	}
	fmt.Printf("Residual parameters: logN=%d, logSlots=%d, H=%d, sigma=%f, logQP=%f, levels=%d, scale=2^%d\n",
		btpParams.ResidualParameters.LogN(),
		btpParams.ResidualParameters.LogMaxSlots(),
		btpParams.ResidualParameters.XsHammingWeight(),
		btpParams.ResidualParameters.Xe(), params.LogQP(),
		btpParams.ResidualParameters.MaxLevel(),
		btpParams.ResidualParameters.LogDefaultScale())
	fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%f, logQP=%f, levels=%d, scale=2^%d\n",
		btpParams.BootstrappingParameters.LogN(),
		btpParams.BootstrappingParameters.LogMaxSlots(),
		btpParams.BootstrappingParameters.XsHammingWeight(),
		btpParams.EphemeralSecretWeight,
		btpParams.BootstrappingParameters.Xe(),
		btpParams.BootstrappingParameters.LogQP(),
		btpParams.BootstrappingParameters.QCount(),
		btpParams.BootstrappingParameters.LogDefaultScale())

	return llama, helper, size, opeval
}

type TestHelper struct {
	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
	params    *ckks.Parameters
}

func (helper *TestHelper) ctGen(n int) (ctVec []*rlwe.Ciphertext) {
	ctVec = []*rlwe.Ciphertext{}

	for i := 1; i <= n; i++ {
		message := make([]float64, helper.params.MaxSlots())
		for i := range message {
			if i % 4 == 0 {
				message[i] = sampling.RandFloat64(-10, 10)
			} else {
				message[i] = 0.0
			}
		}
		plaintext := ckks.NewPlaintext(*(helper.params), *level)
		if err := helper.encoder.Encode(message, plaintext); err != nil {
			panic(err)
		}
		ciphertext, err := helper.encryptor.EncryptNew(plaintext)
		if err != nil {
			panic(err)
		}
		ctVec = append(ctVec, ciphertext)
	}

	return
}

func (helper *TestHelper) constCtGen(c float64, level int) (ct *rlwe.Ciphertext) {
	message := make([]float64, helper.params.MaxSlots())
	for i := range message {
		message[i] = c
	}
	plaintext := ckks.NewPlaintext(*(helper.params), level)
	helper.encoder.Encode(message, plaintext)
	ct, _ = helper.encryptor.EncryptNew(plaintext)

	return ct
}

func (helper *TestHelper) ptGen(n int) (ptVec []*rlwe.Plaintext) {
	ptVec = []*rlwe.Plaintext{}

	for i := 1; i <= n; i++ {
		message := make([]complex128, helper.params.MaxSlots())
		for i := range message {
			message[i] = sampling.RandComplex128(-1, 1)
		}
		plaintext := ckks.NewPlaintext(*(helper.params), *level)
		if err := helper.encoder.Encode(message, plaintext); err != nil {
			panic(err)
		}
		ptVec = append(ptVec, plaintext)
	}

	return
}

func (helper *TestHelper) constPtGen(c float64, level int) (pt *rlwe.Plaintext) {
	message := make([]float64, helper.params.MaxSlots())
	for i := range message {
		message[i] = c
	}
	plaintext := ckks.NewPlaintext(*(helper.params), level)
	helper.encoder.Encode(message, plaintext)
	pt = plaintext

	return pt
}

// IsInteger checks if a float64 value is actually an integer (within floating point precision)
// This is used to optimize CKKS operations: integer constants don't increase scale
func IsInteger(f float64) bool {
	return f == float64(int(f)) || math.Abs(f-math.Trunc(f)) < 1e-10
}

// IsIntegerBig checks if a *big.Float value is actually an integer
// Uses big.Float's built-in IsInt() method for exact determination
// Used to optimize CKKS operations: integer constants don't increase scale
func IsIntegerBig(f *big.Float) bool {
	if f == nil {
		return false
	}
	return f.IsInt()
}

func (helper *TestHelper) idxPtGen() (ptVec []*rlwe.Plaintext) {
	ptVec = []*rlwe.Plaintext{}
	message := make([]complex128, helper.params.MaxSlots())

	for i := range message {
		message[i] = complex(float64((i / 8) % 16) - 8, float64((i / 8) / 16) - 8)
	}
	plaintext := ckks.NewPlaintext(*(helper.params), *level)
	if err := helper.encoder.Encode(message, plaintext); err != nil {
		panic(err)
	}
	ptVec = append(ptVec, plaintext)

	return
}

func (helper *TestHelper) Dec(ciphertext *rlwe.Ciphertext, n int) []complex128 {
	msg := make([]complex128, ciphertext.Slots())

	if err := helper.encoder.Decode(helper.decryptor.DecryptNew(ciphertext), msg); err != nil {
		panic(err)
	}
	if n > 0 {
		fmt.Printf("Level: %d, LogScale: %f\n", ciphertext.Level(), math.Log2(ciphertext.Scale.Float64()))
		// fmt.Printf("First %d elem of msg:\n", n)
	}
	for i := 0; i < n; i++ {
		// fmt.Printf("%.2f+%.2fi; ", real(msg[i]), imag(msg[i]))
		fmt.Printf("%.4f; ", real(msg[i]))
	}
	fmt.Printf("\n")
	
	return msg
}

func (helper *TestHelper) MSE(msg1 []complex128, msg2 []complex128) (d float64) {
	fmt.Print("Computing MSE...\n")
	var sum float64
	for i := range msg1 {
		sum += (real(msg1[i]) - real(msg2[i])) * (real(msg1[i]) - real(msg2[i]))
	}
	err := math.Sqrt(sum / float64(len(msg1)))
	for i := 0; i < 10; i++ {
		fmt.Printf("%.4f; ", real(msg1[i]))
	}
	fmt.Print("\n")
	for i := 0; i < 10; i++ {
		fmt.Printf("%.4f; ", real(msg2[i]))
	}
	fmt.Print("\n")
	fmt.Printf("Error: %f\n", err)
	fmt.Printf("Precision: %.2f bits\n", -math.Log2(err))
	return 
}

// Neg negates a ciphertext in-place without changing its scale
// This is more efficient than multiplying by -1 as it avoids rescaling
func (helper *TestHelper) Neg(eval *ckks.Evaluator, ct *rlwe.Ciphertext) {
	level := ct.Level()
	ringQ := helper.params.RingQ().AtLevel(level)
	for i := range ct.Value {
		ringQ.Neg(ct.Value[i], ct.Value[i])
	}
}

func (helper *TestHelper) encodePlaintext(weights [][]complex128) (ptVec []*rlwe.Plaintext) {
	ptVec = make([]*rlwe.Plaintext, len(weights))

	if *parallel {
		var wg sync.WaitGroup
		numThreads := runtime.GOMAXPROCS(0)
		chunkSize := (len(weights) + numThreads - 1) / numThreads
		for t := 0; t < numThreads; t++ {
			startIdx := t * chunkSize
			endIdx := (t + 1) * chunkSize
			if endIdx > len(weights) {
				endIdx = len(weights)
			}
			if startIdx >= len(weights) {
				continue
			}
			wg.Add(1)
			go func(startIdx, endIdx int, tID int) {
				fmt.Printf("Thread %d starting encode from %d to %d\n", tID, startIdx, endIdx)
				defer wg.Done()
				localEncoder := ckks.NewEncoder(*(helper.params))
				for i := startIdx; i < endIdx; i++ {
					plaintext := ckks.NewPlaintext(*(helper.params), 13)
					if err := localEncoder.Encode(weights[i], plaintext); err != nil {
						panic(err)
					}
					ptVec[i] = plaintext
					if (i-startIdx)%100 == 0 {
						fmt.Printf("Thread %d encoded %d/%d\n", tID, i-startIdx, endIdx-startIdx)
					}
				}
				fmt.Printf("Thread %d completed encode.\n", tID)
			}(startIdx, endIdx, t)
		}
		wg.Wait()
	} else {
		for i, weight := range weights {
			plaintext := ckks.NewPlaintext(*(helper.params), 13)
			if err := helper.encoder.Encode(weight, plaintext); err != nil {
				panic(err)
			}
			ptVec[i] = plaintext
		}
	}

	return
}

func (helper *TestHelper) encryptCiphertext(weights [][]complex128) (ctVec []*rlwe.Ciphertext) {
	ctVec = []*rlwe.Ciphertext{}

	for _, weight := range weights {
		plaintext := ckks.NewPlaintext(*(helper.params), *level)
		if err := helper.encoder.Encode(weight, plaintext); err != nil {
			panic(err)
		}
		ciphertext, _ := helper.encryptor.EncryptNew(plaintext)

		ctVec = append(ctVec, ciphertext)
	}

	return
}

func (helper *TestHelper) PrepareWeights(size *LlamaSize, weights []string, llama *LlamaInference) {
    if llama.wMsg == nil {
        llama.wMsg = make(map[string][][]complex128)
    }

	fmt.Print("Preparing weights...\n")
	for _, name := range weights {
		switch name {
		case "q", "k", "v", "out":
			helper.readWeightsLinear(name, size, llama, 0, "")
		case "up", "gate":
			helper.readWeightsLinear(name, size, llama, 1, "")
		case "down":
			helper.readWeightsLinear(name, size, llama, -1, "")
		case "RoPE":
			helper.prepareRoPE(size, llama)
		default:
			panic("Weight name not recognized!")
		}
	}
	fmt.Print("Weights prepared!\n")
}

func readWeightsFromFile(size *LlamaSize, expand int) (weights [][]complex128) {
	// In actual implementation, this function would read weights from a file.
	// Here we return a dummy weight matrix for demonstration purposes.

	weights = [][]complex128{}
	dim1, dim2 := size.hidDim, size.hidDim
	if expand < 0 {
		dim2 = size.expDim
	} else if expand > 0 {
		dim1 = size.expDim
	}
	for i := 0; i < dim1; i++ {
		row := make([]complex128, dim2)
		for j := 0; j < dim2; j++ {
			row[j] = complex(0.01, 0)
		}
		weights = append(weights, row)
	}
	return
}

// Core function for weight encoding
func (helper *TestHelper) readWeightsLinear(name string, size *LlamaSize, llama *LlamaInference, expand int, filePath string) {
	hidDim := size.hidDim
	expDim := size.expDim
	numSlots := llama.params.MaxSlots()
	fmt.Printf("Generating weights matrix for %s...\n", name)
	weightMatrix := readWeightsFromFile(size, expand)
	weightPoly := make([][]complex128, 0)
	var batch, inRot, outRot int

	if expand == 0 {
		inRot = int(math.Sqrt(float64(hidDim * hidDim / (2 * numSlots))))
		outRot = hidDim * hidDim / (numSlots * inRot)
		batch = numSlots / hidDim
	} else {
		inRot = int(math.Sqrt(float64(hidDim * expDim / (2 * numSlots))))
		for (hidDim * expDim / (2 * numSlots)) % inRot != 0 {
			inRot--
		}
		outRot = hidDim * expDim / (numSlots * inRot)
		batch = numSlots / expDim
	}

	// i, j and k determine the diagonal index
	for i := 0; i < outRot; i++ {
		for j := 0; j < inRot; j++ {
			poly := make([]complex128, numSlots)
			for k := 0; k < batch; k++ {
				// l determines the position within the diagonal
				if expand == 0 {
					for l := 0; l < hidDim; l++ {
						idx := (k + batch * (l + i * inRot * batch)) % numSlots
						poly[idx] = weightMatrix[l][(k + j * batch + i * inRot * batch + l) % hidDim]
					}
				} else if expand > 0 {
					for l := 0; l < expDim; l++ {
						// The output follows the order 0, d, 2d, ..., 1, d+1, ..., (a-1)d, ..., ad-1
						idx := (k + l / hidDim * batch + batch * (expDim / hidDim) * (l % hidDim + i * inRot * batch)) % numSlots
						poly[idx] = weightMatrix[(l + l / hidDim * batch) % expDim][(k + j * batch + i * inRot * batch + l + l / hidDim * batch) % hidDim]
					}
				} else {
					for l := 0; l < expDim; l++ {
						idx := (k + l / hidDim * batch + batch * (expDim / hidDim) * (l % hidDim + i * inRot * batch)) % numSlots
						poly[idx] = weightMatrix[(l + l / hidDim * batch) % hidDim][(k + j * batch + i * inRot * batch + l + l / hidDim * batch) % expDim]
					}
				}
			}
			weightPoly = append(weightPoly, poly)
		}
	}

	fmt.Printf("Matrix generated. Encoding plaintext for %s (size %d)...\n", name, len(weightPoly))
	llama.wMsg[name] = weightMatrix
	llama.w[name] = helper.encodePlaintext(weightPoly)
	fmt.Printf("Finished encoding for %s\n", name)
}

func (helper *TestHelper) prepareRoPE(size *LlamaSize, llama *LlamaInference) {
	hidDim := size.hidDim
	seqLen := size.seqLen
	numSlot := helper.params.MaxSlots()
	
	cos := make([]complex128, numSlot)
	sin0 := make([]complex128, numSlot)
	sin1 := make([]complex128, numSlot)
	for i := 0; i < hidDim / 2; i++ {
		theta := float64(seqLen) * (1.0 / math.Pow(10000, float64(2*i)/float64(hidDim)))
		cos[2 * i * numSlot / hidDim] = complex(math.Cos(theta), 0)
		sin0[2 * i * numSlot / hidDim] = complex(math.Sin(theta), 0)
		cos[(2 * i + 1) * numSlot / hidDim] = complex(math.Cos(theta), 0)
		sin1[(2 * i + 1) * numSlot / hidDim] = -complex(math.Sin(theta), 0)
	}

	llama.w["RoPE"] = []*rlwe.Plaintext{}
	cosPt := ckks.NewPlaintext(*(helper.params), *level)
	helper.encoder.Encode(cos, cosPt)
	llama.w["RoPE"] = append(llama.w["RoPE"], cosPt)
	sin0Pt := ckks.NewPlaintext(*(helper.params), *level)
	helper.encoder.Encode(sin0, sin0Pt)
	llama.w["RoPE"] = append(llama.w["RoPE"], sin0Pt)
	sin1Pt := ckks.NewPlaintext(*(helper.params), *level)
	helper.encoder.Encode(sin1, sin1Pt)
	llama.w["RoPE"] = append(llama.w["RoPE"], sin1Pt)
}

func (helper *TestHelper) PrepareCache(size *LlamaSize, cacheList []string, llama *LlamaInference) {
	llama.cacheMsg = make(map[string][][]complex128)

	fmt.Print("Preparing cache...\n")
	for _, name := range cacheList {
		switch name {
		case "k":
			helper.readCache("k", size, llama)
		case "v":
			helper.readCache("v", size, llama)
		default:
			panic("Cache name not recognized!")
		}
	}
	fmt.Print("Cache prepared!\n")
}

func genCache(size *LlamaSize) (weights [][]complex128) {
	// In actual implementation, the cache should be generated during prefilling stage.
	// Here we return a dummy weight matrix for demonstration purposes.

	weights = [][]complex128{}
	for i := 0; i < size.seqLen; i++ {
		row := make([]complex128, size.hidDim)
		for j := 0; j < size.hidDim; j++ {
			row[j] = complex(0.01, 0)
		}
		weights = append(weights, row)
	}
	return
}

func (helper *TestHelper) readCache(name string, size *LlamaSize, llama *LlamaInference) {
	hidDim := size.hidDim
	seqLen := size.seqLen
	numHeads := size.numHeads
	numSlots := llama.params.MaxSlots()
	cacheMatrix := genCache(size)
	cachePoly := make([][]complex128, 0)
	maskPoly := make([][]complex128, 0)

	if name == "k" {
		batch := numSlots / hidDim
		cacheLen := hidDim / numHeads
		// Extend K cache length to support seqLen >= cacheLen * batch
		if needed := (seqLen + batch) / batch; needed > cacheLen {
			cacheLen = needed
		}
		maskPeriod := numHeads * numSlots / hidDim  // mask pattern period
		for i := 0; i < cacheLen; i++ {
			poly := make([]complex128, numSlots)
			mask := make([]complex128, numSlots)

			if i < (seqLen + numSlots / hidDim - 1) * hidDim / numSlots {
				for j := 0; j < numSlots / hidDim; j++ {
					for k := 0; k < hidDim; k++ {
						if j + i * numSlots / hidDim < len(cacheMatrix) {
							poly[j + k * numSlots / hidDim] = cacheMatrix[j + i * numSlots / hidDim][k]
						}
					}
				}
			} else {
				for j := 0; j < numSlots; j++ {
					poly[j] = 0
				}
			}
			// Mask wraps with period maskPeriod for extended cache indices
			maskOffset := (i % (hidDim / numHeads)) * maskPeriod
			for j := 0; j < numSlots; j++ {
				if j >= maskOffset && j < maskOffset + maskPeriod {
					mask[j] = 1
				} else {
					mask[j] = 0
				}
			}

			cachePoly = append(cachePoly, poly)
			maskPoly = append(maskPoly, mask)
		}

		llama.cache["k"] = helper.encryptCiphertext(cachePoly)
		llama.mask["k"] = helper.encodePlaintext(maskPoly)
		llama.cacheMsg["k"] = cacheMatrix
	} else if name == "v" {
		cacheLen := hidDim / numHeads
		inRot := int(math.Sqrt(float64(cacheLen)))
		for cacheLen%inRot != 0 {
			inRot--
		}
		outRot := cacheLen / inRot
		
		// i and j determine the "block diagonal" index
		for i := 0; i < outRot; i++ {
			for j := 0; j < inRot; j++ {
				poly := make([]complex128, numSlots)
				// k and l determine the position within one block
				for k := 0; k < numHeads; k++ {
					for l := 0; l < numSlots / hidDim; l++ {
						// m determines the block in the diagonal
						for m := 0; m < cacheLen; m++ {
							idx := (l + k * numSlots / hidDim + m * numSlots * numHeads / hidDim) % numSlots
							row := ((m + j) * numSlots / hidDim + l) % (numSlots / numHeads)
							col := (((hidDim - numHeads + m - i * inRot) % (hidDim - numHeads)) * numHeads + k) % hidDim
							if row < seqLen {
								poly[idx] = cacheMatrix[row][col]
							} else {
								poly[idx] = 0
							}
						}
					}
				}
				cachePoly = append(cachePoly, poly)
			}
		}

		mask := make([]complex128, numSlots)
		for i := 0; i < numSlots; i++ {
			if i % (numSlots / hidDim) == 0 {
				mask[i] = 1
			} else {
				mask[i] = 0
			}
		}
		maskPoly = append(maskPoly, mask)

		llama.cache["v"] = helper.encryptCiphertext(cachePoly)
		llama.mask["v"] = helper.encodePlaintext(maskPoly)
		llama.cacheMsg["v"] = cacheMatrix
	}
}

type OperationEvaluator struct {
	helper  *TestHelper
	eval    *ckks.Evaluator
	btpEval *bootstrapping.Evaluator
	ctVec   []*rlwe.Ciphertext
	ptVec   []*rlwe.Plaintext
}

func NewOperationEvaluator(helper *TestHelper, eval *ckks.Evaluator, btpEval *bootstrapping.Evaluator, n int) *OperationEvaluator {
	return &OperationEvaluator{
		helper:  helper,
		eval:    eval,
		btpEval: btpEval,
		ctVec:   helper.ctGen(n),
		ptVec:   helper.ptGen(n),
	}
}

func (opeval *OperationEvaluator) add() {
	ctVec := opeval.ctVec
	var err error

	start := time.Now()
	for i := range ctVec {
		_, err = opeval.eval.AddNew(ctVec[i], ctVec[i])
		if err != nil {
			panic(err)
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Addition Consumed %f seconds with level %d\n", elapsed.Seconds() / float64(len(ctVec)), *level)
}

func (opeval *OperationEvaluator) ctPtMult() {
	ctVec := opeval.ctVec
	ptVec := opeval.ptVec
	var ctTmp *rlwe.Ciphertext
	var err error

	start := time.Now()
	ctTmp = ctVec[0]
	for i := range ctVec {
		_, err = opeval.eval.MulNew(ctVec[i], ptVec[i])
		if err != nil {
			panic(err)
		}
		opeval.eval.Rescale(ctVec[i], ctTmp)
	}
	elapsed := time.Since(start)
	fmt.Printf("Ct-pt multiplication Consumed %f seconds with level %d\n", elapsed.Seconds() / float64(len(ctVec)), *level)
}

func (opeval *OperationEvaluator) ctCtMult() {
	ctVec := opeval.ctVec
	var ctTmp *rlwe.Ciphertext
	var err error

	start := time.Now()
	ctTmp = ctVec[0]
	for i := range ctVec {
		_, err = opeval.eval.MulRelinNew(ctVec[i], ctVec[i])
		if err != nil {
			panic(err)
		}
		opeval.eval.Rescale(ctVec[i], ctTmp)
	}
	elapsed := time.Since(start)
	fmt.Printf("Ct-ct multiplication Consumed %f seconds with level %d\n", elapsed.Seconds() / float64(len(ctVec)), *level)
}

func (opeval *OperationEvaluator) rotate() {
	ctVec := opeval.ctVec

	start := time.Now()
	for i := range ctVec {
		opeval.eval.RotateNew(ctVec[i], 5)
	}
	elapsed := time.Since(start)
	fmt.Printf("Rotation Consumed %f seconds with level %d\n", elapsed.Seconds() / float64(len(ctVec)), *level)

	start = time.Now()
	nums := slices.Repeat([]int{5}, len(ctVec))
	opeval.eval.RotateHoistedNew(ctVec[0], nums)
	elapsed = time.Since(start)
	fmt.Printf("Hoisted rotation Consumed %f seconds with level %d\n", elapsed.Seconds() / float64(len(ctVec)), *level)
}

func (opeval *OperationEvaluator) drop() {
	ctVec := opeval.ctVec

	start := time.Now()
	for i := range ctVec {
		opeval.eval.DropLevelNew(ctVec[i], 1)
	}
	elapsed := time.Since(start)
	fmt.Printf("Level drop Consumed %f seconds with level %d\n", elapsed.Seconds() / float64(len(ctVec)), *level)
}

func (opeval *OperationEvaluator) boot() {
	ctVec := opeval.ctVec

	start := time.Now()
	opeval.btpEval.Bootstrap(ctVec[0])
	elapsed := time.Since(start)
	fmt.Printf("Bootstrapping Consumed %f seconds with level %d\n", elapsed.Seconds(), *level)
}

func (opeval *OperationEvaluator) EvaluateAll() {
	fmt.Printf("Evaluating All Operations\n")

	opeval.add()
	opeval.ctCtMult()
	opeval.ctPtMult()
	opeval.rotate()
	opeval.drop()
	opeval.boot()
}
