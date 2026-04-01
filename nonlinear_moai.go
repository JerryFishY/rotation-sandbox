package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func (llama *LlamaInference) SiLU(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	if llama.siluFunc == nil {
		fmt.Print("Preparing SiLU polynomial approximation... This can be amortized.\n")
		silu := func(x float64) (y float64) {
			return x / (math.Exp(-x) + 1)
		}
		interval := bignum.Interval{A: *bignum.NewFloat(-20, 128), B: *bignum.NewFloat(20, 128), Nodes: 1023}
		nonZero := make([]int, llama.size.expDim)
		for i := 0; i < llama.size.expDim; i++ {
			nonZero[i] = i * llama.params.MaxSlots() / llama.size.expDim
		}
		mapping := map[int][]int{0: nonZero}
		poly, _ := polynomial.NewPolynomialVector([]bignum.Polynomial{bignum.ChebyshevApproximation(silu, interval)}, mapping)
		polyeval := polynomial.NewEvaluator(*llama.params, llama.eval[0])
		scale, constant := poly.ChangeOfBasis(llama.params.MaxSlots())
		llama.siluFunc = &SiluFunc{
			siluEval: polyeval,
			siluPoly: poly,
			scalar:   scale[0],
			constant: constant[0],
		}
	}

	fmt.Print("Computing SiLU...\n")
	start := time.Now()
	var err error
	llama.eval[0].Mul(x, llama.siluFunc.scalar, x)
	llama.eval[0].Add(x, llama.siluFunc.constant, x)
	llama.eval[0].Rescale(x, x)
	y, err = llama.siluFunc.siluEval.Evaluate(x, llama.siluFunc.siluPoly, llama.params.DefaultScale())
	if err != nil {
		panic(err)
	}
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), x.Level(), y.Level())
	return y
}

func (llama *LlamaInference) SiLUPlaintext(x []complex128) []complex128 {
	y := make([]complex128, len(x))

	for i, val := range(x) {
		y[i] = complex(real(val) / (math.Exp(-real(val)) + 1), 0)
	}
	return y
}

func (llama *LlamaInference) Softmax(x *rlwe.Ciphertext, btpLevel int, temp int) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	fmt.Print("Computing Softmax...\n")

	// evaluating exp
	if x.Level() < 8 {
		panic("Not enough levels for exp.")
	}
	start := time.Now()
	inLevel := x.Level()
	inv128 := llama.helper.constPtGen(0.0078125, x.Level())
	two := llama.helper.constPtGen(-2.5, x.Level())
	eval.Add(x, two, x)
	err := eval.Mul(x, inv128, x)
	if err != nil {
		panic(err)
	}
	eval.Rescale(x, x)
	one := llama.helper.constPtGen(1, x.Level())
	eval.Add(x, one, x)
	for i := 0; i < (7 + temp); i++ {
		err = eval.MulRelin(x, x, x)
		if err != nil {
			panic(err)
		}
		eval.Rescale(x, x)
	}
	exp := x.CopyNew()

	// evaluating sum
	for j := 1; j < 256; j *= 2 {
		tmp := x.CopyNew()
		// eval.Rotate(tmp, 1024 / j, tmp)
		rotateAnyStep(eval, x, 1024 / j, tmp)
		eval.Add(x, tmp, x)
	}
	elapsed := time.Since(start)
	midLevel := x.Level()
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), inLevel, midLevel)

	// evaluating inverse
	if btpLevel <= 13 {
		panic("Not enough levels for inverse.")
	}
	x = llama.BootTo(x, btpLevel)
	scale := llama.helper.constPtGen(0.1, x.Level())
	eval.Mul(x, scale, x)
	eval.Rescale(x, x)
	start = time.Now()
	oneCt := llama.helper.constCtGen(1, x.Level())
	res, _ := eval.SubNew(oneCt, x)
	dnm, _ := eval.AddNew(oneCt, res)
	for i := 0; i < 9; i++ {
		err = eval.MulRelin(res, res, res)
		if err != nil {
			panic(err)
		}
		eval.Rescale(res, res)
		eval.DropLevel(oneCt, 1)
		tmp, _ := eval.AddNew(res, oneCt)
		err = eval.MulRelin(dnm, tmp, dnm)
		if err != nil {
			panic(err)
		}
		eval.Rescale(dnm, dnm)
	}

	y, err = eval.MulRelinNew(exp, dnm)
	if err != nil {
		panic(err)
	}
	eval.Rescale(y, y)
	scale = llama.helper.constPtGen(0.1, y.Level())
	eval.Mul(y, scale, y)
	eval.Rescale(y, y)

	elapsed = time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), btpLevel, y.Level())

	return
}

func (llama *LlamaInference) SoftmaxPlaintext(x []complex128) []complex128 {
	y := make([]complex128, len(x))
	exp := make([]float64, len(x))
	sum := make([]float64, 8)

	for i, val := range(x) {
		exp[i] = math.Exp(real(val))
		sum[i % 8] += exp[i]
	}
	for i, val := range(exp) {
		y[i] = complex(val / sum[i % 8], 0)
	}

	return y
}

func (llama *LlamaInference) Norm(x *rlwe.Ciphertext, btpLevel int) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	fmt.Print("Computing Norm...\n")
	inLevel := x.Level()
	if inLevel < 13 {
		panic("Not enough levels for variance.")
	}
	start := time.Now()

	// computing mean
	mean := x.CopyNew()
	for i := llama.params.MaxSlots() / llama.size.hidDim; i < llama.params.MaxSlots(); i = i * 2 {
		tmp := mean.CopyNew()
		// eval.Rotate(tmp, i, tmp)
		rotateAnyStep(eval, mean, i, tmp)
		eval.Add(tmp, mean, mean)
	}

	// computing difference between d*x and mean
	d := llama.helper.constPtGen(float64(llama.size.hidDim), x.Level())
	xd := x.CopyNew()
	eval.Mul(x, d, xd)
	eval.Rescale(xd, xd)
	eval.DropLevel(mean, 1)
	varc, _ := eval.SubNew(xd, mean)

	// computing variance
	eval.MulRelin(varc, varc, varc)
	eval.Rescale(varc, varc)
	for i := llama.params.MaxSlots() / llama.size.hidDim; i < llama.params.MaxSlots(); i = i * 2 {
		tmp := varc.CopyNew()
		// eval.Rotate(tmp, i, tmp)
		rotateAnyStep(eval, varc, i, tmp)
		eval.Add(tmp, varc, varc)
	}

	// normalizing variance
	dInvCube := llama.helper.constPtGen(1/math.Pow(float64(llama.size.hidDim), 3), varc.Level())
	eval.Mul(varc, dInvCube, varc)
	eval.Rescale(varc, varc)

	// evaluating inv sqrt
	NewtonIter := func(x *rlwe.Ciphertext, ans *rlwe.Ciphertext, it int) (y *rlwe.Ciphertext) {
		threeHalf := llama.helper.constPtGen(1.5, x.Level())
		negHalf := llama.helper.constPtGen(-0.5, x.Level())
		halfX, _ := eval.MulNew(x, negHalf)
		eval.Rescale(halfX, halfX)

		for i := 0; i < it; i++ {
			ansSquare, _ := eval.MulRelinNew(ans, ans)
			eval.Rescale(ansSquare, ansSquare)
			eval.DropLevel(ans, 1)
			ansCube, _ := eval.MulRelinNew(ansSquare, ans)
			eval.Rescale(ansCube, ansCube)
			for ansCube.Level() < halfX.Level() {
				eval.DropLevel(halfX, 1)
			}
			term1, _ := eval.MulRelinNew(ansCube, halfX)
			eval.Rescale(term1, term1)
			term2, _ := eval.MulRelinNew(ans, threeHalf)
			eval.Rescale(term2, term2)
			eval.DropLevel(term2, term2.Level()-term1.Level())
			eval.Add(term1, term2, ans)
		}

		y = ans
		return
	}

	GoldIter := func(x *rlwe.Ciphertext, ans *rlwe.Ciphertext, it int) (y *rlwe.Ciphertext) {
		if x.Level() > ans.Level() {
			eval.DropLevel(x, x.Level()-ans.Level())
		} else if ans.Level() > x.Level() {
			eval.DropLevel(ans, ans.Level()-x.Level())
		}
		sqrt, _ := eval.MulRelinNew(x, ans)
		eval.Rescale(sqrt, sqrt)

		for i := 0; i < it; i++ {
			res, _ := eval.MulRelinNew(sqrt, ans)
			eval.Rescale(res, res)
			two := llama.helper.constPtGen(2, res.Level())
			eval.Sub(res, two, res)
			eval.DropLevel(sqrt, 1)
			eval.DropLevel(ans, 1)
			eval.MulRelin(sqrt, res, sqrt)
			eval.Rescale(sqrt, sqrt)
			eval.MulRelin(ans, res, ans)
			eval.Rescale(ans, ans)
		}

		y = ans
		return
	}

	// initial approximation
	slope := llama.helper.constPtGen(-1.29054537e-04, varc.Level())
	bias := llama.helper.constPtGen(1.29054537e-01, varc.Level()-1)
	ans, _ := eval.MulNew(varc, slope)
	eval.Rescale(ans, ans)
	eval.Add(ans, bias, ans)

	// Newton iterations
	ans = NewtonIter(varc, ans, 4)
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), inLevel, ans.Level())
	
	// Gold iterations
	if btpLevel < 6 {
		panic("Not enough levels for Gold iteration.")
	}
	ans = llama.BootTo(ans, btpLevel)
	start = time.Now()
	ans = GoldIter(varc, ans, 2)

	// final multiplication
	y, _ = eval.MulRelinNew(x, ans)
	eval.Rescale(y, y)
	elapsed = time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), btpLevel, y.Level())
	return
}

func (llama *LlamaInference) NormPlaintext(x []complex128) []complex128 {
	xFloat := make([]float64, len(x))
	varc := make([]float64, len(x))
	result := make([]complex128, len(x))

	average := func (x []float64) []float64 {
		avr := make([]float64, len(x))
		for i := 0; i < len(x) / llama.size.hidDim; i++ {
			for j := 0; j < llama.size.hidDim; j++ {
				avr[i] += x[i + j * len(x) / llama.size.hidDim]
			}
			avr[i] /= float64(llama.size.hidDim)
		}
		for i := len(x) / llama.size.hidDim; i < len(x); i++ {
			avr[i] = avr[i % (len(x) / llama.size.hidDim)]
		}
		return avr
	}

	for i := range x {
		xFloat[i] = real(x[i])
	}
	mean := average(xFloat)
	for i := range x {
		varc[i] = (xFloat[i] - mean[i]) * (xFloat[i] - mean[i])
	}
	varc = average(varc)
	for i := range x {
		result[i] = complex(xFloat[i] / math.Sqrt(varc[i]), 0)
	}

	return result
}

// UNDER CONSTRUCTION
func (llama *LlamaInference) Argmax(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	eval := llama.eval[0]
	logit := llama.Softmax(x, 14, 3)
	copy := logit.CopyNew()

	for j := 1; j < 256; j *= 2 {
		tmp := logit.CopyNew()
		// eval.Rotate(tmp, 1024 / j, tmp)
		rotateAnyStep(eval, logit, 1024 / j, tmp)
		eval.Add(logit, tmp, logit)
	}
	logit = llama.BootTo(logit, 16)
	// scale := llama.helper.constPtGen(2, logit.Level())
	// eval.Mul(logit, scale, logit)
	// eval.Rescale(logit, logit)
	oneCt := llama.helper.constCtGen(1, logit.Level())
	res, _ := eval.SubNew(oneCt, logit)
	dnm, _ := eval.AddNew(oneCt, res)
	for i := 0; i < 12; i++ {
		eval.MulRelin(res, res, res)
		eval.Rescale(res, res)
		eval.DropLevel(oneCt, 1)
		tmp, _ := eval.AddNew(res, oneCt)
		eval.MulRelin(dnm, tmp, dnm)
		eval.Rescale(dnm, dnm)
	}
	y, _ = eval.MulRelinNew(copy, dnm)
	eval.Rescale(y, y)
	// scale = llama.helper.constPtGen(2, y.Level())
	// eval.Mul(y, scale, y)
	// eval.Rescale(y, y)

	// llama.helper.Dec(copy, 128)
	// llama.helper.Dec(y, 128)
	idx := llama.helper.idxPtGen()[0]
	eval.MulRelin(y, idx, y)
	eval.Rescale(y, y)
	for i := 1; i < 256; i *= 2 {
		tmp := y.CopyNew()
		// eval.Rotate(tmp, 1024 / i, tmp)
		rotateAnyStep(eval, y, 1024 / i, tmp)
		eval.Add(y, tmp, y)
	}

	return
}

// UNDER CONSTRUCTION
func (llama *LlamaInference) ArgmaxPlaintext(x []complex128) []int {
	idx := make([]int, 8)
	for j := 0; j < 8; j++ {
		idx[j] = -1
		max := math.Inf(-1)
		for i, val := range x {
			if real(val) > max && i % 8 == j {
				max = real(val)
				idx[j] = i / 8
			}
		}
	}

	return idx
}

func (llama *LlamaInference) BootTo(x *rlwe.Ciphertext, l int) (y *rlwe.Ciphertext) {
	if llama.btpEval.OutputLevel() < l {
		panic("Required output level is larger than the maximum!")
	}
	inLevel := x.Level()
	fmt.Printf("Bootstrapping called...\n")
	start := time.Now()
	y, _ = llama.btpEval.Bootstrap(x)
	if l < x.Level() {
		llama.eval[0].DropLevel(x, x.Level()-l)
	}
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), inLevel, l)
	return y
}
