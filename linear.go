package main

import (
	"fmt"
	"math"
	"time"
	"sync"
	"runtime"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func (llama *LlamaInference) Linear(xPtr *rlwe.Ciphertext, wStr string, expand int) (y *rlwe.Ciphertext) {
	eval := llama.eval

	var preProc, postProc, inRot, outRot, rotStep int
	numSlots := llama.params.MaxSlots()
	hidDim := llama.size.hidDim
	expDim := llama.size.expDim
	if expand >= 0 {
		preProc = numSlots / hidDim
	} else {
		preProc = numSlots / expDim
	}
	if expand <= 0 {
		postProc = numSlots / hidDim
		rotStep = preProc * preProc
	} else {
		postProc = numSlots / expDim
		rotStep = preProc * postProc
	}
	if expand == 0 {
		inRot = int(math.Sqrt(float64(hidDim * hidDim / (2 * numSlots))))
		outRot = hidDim * hidDim / (numSlots * inRot)
	} else {
		inRot = int(math.Sqrt(float64(hidDim * expDim / (2 * numSlots))))
		for (hidDim * expDim / (2 * numSlots)) % inRot != 0 {
			inRot--
		}
		outRot = hidDim * expDim / (numSlots * inRot)
	}

	weight := llama.w[wStr]
	x := xPtr.CopyNew()
	ctRot := make([]*rlwe.Ciphertext, inRot)
	partSum := make([]*rlwe.Ciphertext, len(weight))
	// fmt.Printf("preProc = %d, input_rot = %d, output_rot = %d, mult = %d\n", preProc, inRot, outRot, len(weight))

	for i := 1; i < preProc; i *= 2 { // inter rot
		tmp := x.CopyNew()
		// eval[0].Rotate(x, i * (preProc - 1), tmp)
		rotateAnyStep(eval[0], x, i * (preProc - 1), tmp)
		eval[0].Add(x, tmp, x)
	}
	
	for i := 0; i < inRot; i++ { // input rot
		ctRot[i] = x.CopyNew()
		rotateAnyStep(eval[0], x, rotStep, x)
	}


	if *parallel {
		var wg sync.WaitGroup
		numThreads := runtime.GOMAXPROCS(0)
		if numThreads > inRot {
			numThreads = inRot
		}
		sem := make(chan struct{}, numThreads)
		chunkSize := (len(weight) + numThreads - 1) / numThreads
		if chunkSize%inRot != 0 {
			chunkSize = ((chunkSize + inRot - 1) / inRot) * inRot
		}
		for t := 0; t < numThreads; t++ {
			startIdx := t * chunkSize
			endIdx := (t + 1) * chunkSize
			if endIdx > len(weight) {
				endIdx = len(weight)
			}
			
			wg.Add(1)
			sem <- struct{}{}
			go func(startIdx, endIdx, tid int) {
				defer wg.Done()
				defer func() { <-sem }()
				localEval := eval[tid%runtime.GOMAXPROCS(0)]
				for i := startIdx; i < endIdx; i++ { // mult
					partSum[i], _ = localEval.MulRelinNew(ctRot[i%inRot], weight[i])
					localEval.Rescale(partSum[i], partSum[i])
					if i%inRot > 0 { // input sum
						localEval.Add(partSum[i-i%inRot], partSum[i], partSum[i-i%inRot])
					}
				}
			}(startIdx, endIdx, t)
		}
		wg.Wait()
	} else {
		for i := 0; i < len(weight); i++ { // mult
			partSum[i], _ = eval[0].MulRelinNew(ctRot[i%inRot], weight[i])
			eval[0].Rescale(partSum[i], partSum[i])
			if i%inRot > 0 { // input sum
				eval[0].Add(partSum[i-i%inRot], partSum[i], partSum[i-i%inRot])
			}
		}
	}

	if *parallel {
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.GOMAXPROCS(0))
		for i := 1; i < outRot; i++ { // output rot
			wg.Add(1)
			sem <- struct{}{} 
			go func(i int) {
				defer wg.Done()
				defer func() { <-sem }()
				localEval := eval[i%runtime.GOMAXPROCS(0)]
				// localEval.Rotate(partSum[i*inRot], i * rotStep * inRot, partSum[i*inRot])
				rotateAnyStep(localEval, partSum[i*inRot], i * rotStep * inRot, partSum[i*inRot])
			}(i)
		}
		wg.Wait()
	} else {
		for i := 1; i < outRot; i++ { // output rot
			// eval[0].Rotate(partSum[i*inRot], i * rotStep * inRot, partSum[i*inRot])
			rotateAnyStep(eval[0], partSum[i*inRot], i * rotStep * inRot, partSum[i*inRot])
		}
	}

	for i := 1; i < outRot; i++ { // output sum
		eval[0].Add(partSum[0], partSum[i*inRot], partSum[0])
	}

	for i := 1; i < postProc; i *= 2 { // inter sum
		tmp := partSum[0].CopyNew()
		// eval[0].Rotate(partSum[0], i, tmp)
		rotateAnyStep(eval[0], partSum[0], i, tmp)
		eval[0].Add(partSum[0], tmp, partSum[0])
	}
	y = partSum[0]

	return y
}

func (llama *LlamaInference) LinearMsg(x []complex128, mat [][]complex128, heads ...int) (y []complex128) {
	if len(x) > len(mat[0]) {
		fmt.Printf("WARNING: dimension mismatch: x=%d, expected=%d; auto-adjust applied\n",len(x), len(mat[0]))
		x = x[:len(mat[0])]
	} else if len(x) < len(mat[0]) {
		fmt.Printf("WARNING: dimension mismatch: x=%d, expected=%d; auto-adjust applied\n",len(x), len(mat[0]))
		pad := make([]complex128, len(mat[0]) - len(x))
		x = append(x, pad...)
	}
	n := 1
	if len(heads) > 0 {
		n = heads[0]
	}

    y = make([]complex128, len(mat) * n)
    for i := 0; i < len(mat); i++ {
		for j := 0; j < n; j++ {
        	sum := complex128(0.0)
			for k := 0; k < len(mat[0]) / n; k++ {
				sum += mat[i][k * n + j] * x[k * n + j]
			}
        	y[i * n + j] = sum
		}
    }

    return y
}

func (llama *LlamaInference) QKV(x *rlwe.Ciphertext) (q *rlwe.Ciphertext, k *rlwe.Ciphertext, v *rlwe.Ciphertext) {
	if _, ok := llama.w["q"]; !ok {
		panic("Executing QKV, but w_q not initialized!")
	} else if _, ok := llama.w["k"]; !ok {
		panic("Executing QKV, but w_k not initialized!")
	} else if _, ok := llama.w["v"]; !ok {
		panic("Executing QKV, but w_v not initialized!")
	}

	fmt.Print("Computing QKV...\n")
	start := time.Now()
	q = llama.Linear(x, "q", 0)
	k = llama.Linear(x, "k", 0)
	v = llama.Linear(x, "v", 0)
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), x.Level(), q.Level())

	return q, k, v
}

func (llama *LlamaInference) RoPE(q *rlwe.Ciphertext, k *rlwe.Ciphertext) (yQ *rlwe.Ciphertext, yK *rlwe.Ciphertext) {
	weight, ok := llama.w["RoPE"]
	if !ok {
		panic("Executing RoPE, but w_rope not initialized!")
	}

	fmt.Print("Computing RoPE...\n")
	rope := func(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
		eval := llama.eval[0]
		xCos, _ := eval.MulNew(x, weight[0])
		eval.Rescale(xCos, xCos)
		xSin0, _ := eval.MulNew(x, weight[1])
		eval.Rescale(xSin0, xSin0)
		// eval.Rotate(xSin0, -llama.params.MaxSlots() / *hidDim, xSin0)
		rotateAnyStep(eval, xSin0, -llama.params.MaxSlots() / *hidDim, xSin0)
		xSin1, _ := eval.MulNew(x, weight[2])
		eval.Rescale(xSin1, xSin1)
		// eval.Rotate(xSin1, llama.params.MaxSlots() / *hidDim, xSin1)
		rotateAnyStep(eval, xSin1, llama.params.MaxSlots() / *hidDim, xSin1)
		y, _ = eval.AddNew(xSin0, xSin1)
		eval.Add(xCos, y, y)

		return y
	}

	start := time.Now()
	yQ = rope(q)
	yK = rope(k)
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), q.Level(), yQ.Level())

	return yQ, yK
}

func (llama *LlamaInference) RoPEMsg(q []complex128, k []complex128) (yQ []complex128, yK []complex128) {
	hidDim := llama.size.hidDim
	seqLen := llama.size.seqLen
	yQ = make([]complex128, len(q))
	yK = make([]complex128, len(k))
	for i := 0; i < hidDim / 2; i++ {
		theta := float64(seqLen) * (1.0 / math.Pow(10000, float64(2*i)/float64(hidDim)))
		sin := complex(math.Sin(theta), 0)
		cos := complex(math.Cos(theta), 0)
		yQ[2*i] = q[2*i] * cos - q[2*i+1] * sin
		yQ[2*i+1] = q[2*i] * sin + q[2*i+1] * cos
		yK[2*i] = k[2*i] * cos - k[2*i+1] * sin
		yK[2*i+1] = k[2*i] * sin + k[2*i+1] * cos
	}

	return
}

func (llama *LlamaInference) Cache(k *rlwe.Ciphertext, v *rlwe.Ciphertext) {
	kCache, ok := llama.cache["k"]
	if !ok {
		panic("Executing Caching, but K cache not initialized!")
	}
	vCache, ok := llama.cache["v"]
	if !ok {
		panic("Executing Caching, but V cache not initialized!")
	}
	mask := llama.mask["k"]
	if mask == nil {
		panic("Executing Caching, but cache mask not initialized!")
	}

	fmt.Print("Caching...\n")
	eval := llama.eval[0]
	numSlots := llama.params.MaxSlots()
	hidDim := llama.size.hidDim
	numHeads := llama.size.numHeads
	batch := numSlots / hidDim
	idx1 := llama.size.seqLen % batch
	idx2 := (llama.size.seqLen - idx1) / batch
	vCacheLen := hidDim / numHeads
	inRot := int(math.Sqrt(float64(vCacheLen)))
	for vCacheLen%inRot != 0 {
		inRot--
	}
	outRot := vCacheLen / inRot
	start := time.Now()

	// K cache
	// eval.Rotate(k, idx1, k)
	rotateAnyStep(eval, k, idx1, k)
	eval.Add(kCache[idx2], k, kCache[idx2])
	
	// V cache
	eval.MulRelin(v, llama.mask["v"][0], v)
	eval.Rescale(v, v)
	for i := 0; i < outRot; i++ {
		ctRot := v.CopyNew()
		// eval.Rotate(ctRot, numSlots - i * inRot * numSlots * numHeads / hidDim - idx1, ctRot)
		rotateAnyStep(eval, v, numSlots - i * inRot * numSlots * numHeads / hidDim - idx1, ctRot)
		for j := 0; j < inRot; j++ {
			tmp, _ := eval.MulRelinNew(ctRot, mask[idx2 - j])
			eval.Rescale(tmp, tmp)
			idx0 := (hidDim / numHeads + idx2 - j - i * inRot) % (hidDim / numHeads)
			eval.Add(vCache[idx0], tmp, vCache[idx0])
		}
	}

	llama.cache["k"] = kCache
	llama.cache["v"] = vCache
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds\n", elapsed.Seconds())
}

func (llama *LlamaInference) CacheMsg(k []complex128, v []complex128) {
	llama.cacheMsg["k"] = append(llama.cacheMsg["k"], k)
	llama.cacheMsg["v"] = append(llama.cacheMsg["v"], v)
}

func (llama *LlamaInference) QK_T(q *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	kCache, ok := llama.cache["k"]
	if !ok {
		panic("Executing Q*K^T, but K cache not initialized!")
	}
	mask, ok := llama.mask["k"]
	if !ok {
		panic("Executing Q*K^T, but K mask not initialized!")
	}
	if q.Level() <= 1 {
		panic("Executing Q*K^T requires input level at least 2!")
	}

	fmt.Print("Computing QK^T...\n")
	hidDim := llama.size.hidDim
	numSlots := llama.params.MaxSlots()
	numHeads := llama.size.numHeads
	num_rot := hidDim / llama.size.numHeads

	for i := 1; i < numSlots / hidDim; i *= 2 {
		tmp := q.CopyNew()
		// llama.eval[0].Rotate(q, numSlots - i, tmp)
		rotateAnyStep(llama.eval[0], q, numSlots - i, tmp)
		llama.eval[0].Add(q, tmp, q)
	}

	start := time.Now()
	if *parallel {
		eval := llama.eval
		numThreads := runtime.GOMAXPROCS(0)
		chunkSize := (len(kCache) + numThreads - 1) / numThreads
		var wg sync.WaitGroup
		sem := make(chan struct{}, numThreads)
		results := make([]*rlwe.Ciphertext, len(kCache))

		for t := 0; t < numThreads; t++ {
			startIdx := t * chunkSize
			endIdx := (t + 1) * chunkSize
			if endIdx > len(kCache) {
				endIdx = len(kCache)
			}
			
			wg.Add(1)
			sem <- struct{}{}
			go func(startIdx, endIdx, tid int) {
				defer wg.Done()
				defer func() { <-sem }()
				localEval := eval[tid%len(eval)]

				for i := startIdx; i < endIdx; i++ {
					ctTmp, err := localEval.MulRelinNew(q, kCache[i])
					if err != nil {
						panic(err)
					}
					localEval.Rescale(ctTmp, ctTmp)
					for j := 1; j < num_rot; j *= 2 {
						tmp := ctTmp
						// localEval.Rotate(ctTmp, j * numHeads * numSlots / hidDim, tmp)
						rotateAnyStep(localEval, ctTmp, j * numHeads * numSlots / hidDim, tmp)
						localEval.Add(ctTmp, tmp, ctTmp)
					}
					err = localEval.Mul(ctTmp, mask[i], ctTmp)
					if err != nil {
						panic(err)
					}
					localEval.Rescale(ctTmp, ctTmp)
					results[i] = ctTmp
					// localEval.Rotate(results[i], 5, results[i])
					rotateAnyStep(localEval, results[i], 5, results[i])
				}
			}(startIdx, endIdx, t)
		}
		wg.Wait()

		for i, result := range results {
			if i == 0 {
				y = result
			} else {
				eval[0].Add(y, result, y)
			}
		}
	} else {
		eval := llama.eval[0]
		for i := 0; i < (llama.size.seqLen + numSlots / hidDim - 1) * hidDim / numSlots; i++ {
			ctTmp, err := eval.MulRelinNew(q, kCache[i])
			if err != nil {
				panic(err)
			}
			eval.Rescale(ctTmp, ctTmp)
			for j := 1; j < num_rot; j *= 2 {
				tmp := ctTmp.CopyNew()
				// eval.Rotate(ctTmp, j * numHeads * numSlots / hidDim, tmp)
				rotateAnyStep(eval, ctTmp, j * numHeads * numSlots / hidDim, tmp)
				eval.Add(ctTmp, tmp, ctTmp)
			}
			err = eval.Mul(ctTmp, mask[i], ctTmp)
			if err != nil {
				panic(err)
			}
			eval.Rescale(ctTmp, ctTmp)
			if i == 0 {
				y = ctTmp
			} else {
				eval.Add(y, ctTmp, y)
			}
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), q.Level(), y.Level())

	return y
}

// assume that hidDim / (2 * numHeads) has integer square roots
func (llama *LlamaInference) AttnV(s *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	vCache, ok := llama.cache["v"]
	if !ok {
		panic("Executing Attn*V, but V cache not initialized!")
	}
	mask, ok := llama.mask["v"]
	if !ok {
		panic("Executing Attn*V, but V mask not initialized!")
	}
	if s.Level() <= 1 {
		panic("Executing Attn*V requires input level at least 2!")
	}

	fmt.Print("Computing Attn*V...\n")
	var err error
	numSlots := llama.params.MaxSlots()
	numHeads := llama.size.numHeads
	hidDim := llama.size.hidDim
	eval := llama.eval[0]
	inRot := int(math.Sqrt(float64(hidDim / numHeads)))
	for (hidDim / numHeads)%inRot != 0 {
		inRot--
	}
	outRot := hidDim / (numHeads * inRot)
	rotStep := numSlots * numHeads / hidDim
	partSum := make([]*rlwe.Ciphertext, len(vCache))
	ctRot := make([]*rlwe.Ciphertext, inRot)

	start := time.Now()
	for i := 0; i < inRot; i++ { // input rot
		ctRot[i] = s.CopyNew()
		// eval.Rotate(s, rotStep, s)
		rotateAnyStep(eval, s, rotStep, s)
	}

	if *parallel {
		var wg sync.WaitGroup
		numThreads := runtime.GOMAXPROCS(0)
		if numThreads > inRot {
			numThreads = inRot
		}
		sem := make(chan struct{}, numThreads)
		chunkSize := (len(vCache) + numThreads - 1) / numThreads
		if chunkSize%inRot != 0 {
			chunkSize = ((chunkSize + inRot - 1) / inRot) * inRot
		}
		for t := 0; t < numThreads; t++ {
			startIdx := t * chunkSize
			endIdx := (t + 1) * chunkSize
			if endIdx > len(vCache) {
				endIdx = len(vCache)
			}
			
			wg.Add(1)
			sem <- struct{}{}
			go func(startIdx, endIdx, tid int) {
				defer wg.Done()
				defer func() { <-sem }()
				localEval := llama.eval[t]

				for i := startIdx; i < endIdx; i++ {
					partSum[i], err = localEval.MulRelinNew(ctRot[i%inRot], vCache[i])
					if err != nil {
						panic(err)
					}
					localEval.Rescale(partSum[i], partSum[i])
					if i%inRot > 0 {
						localEval.Add(partSum[i-i%inRot], partSum[i], partSum[i-i%inRot])
					}
				}
			}(startIdx, endIdx, t)
		}
		wg.Wait()
	} else {
		for i := 0; i < len(vCache); i++ {
			partSum[i], err = eval.MulRelinNew(ctRot[i%inRot], vCache[i])
			if err != nil {
				panic(err)
			}
			eval.Rescale(partSum[i], partSum[i])
			if i%inRot > 0 {
				eval.Add(partSum[i-i%inRot], partSum[i], partSum[i-i%inRot])
			}
		}
	}

	for i := 1; i < outRot; i++ {
		// eval.Rotate(partSum[i*inRot], i * rotStep * inRot, partSum[i*inRot])
		rotateAnyStep(eval, partSum[i*inRot], i * rotStep * inRot, partSum[i*inRot])
		eval.Add(partSum[0], partSum[i*inRot], partSum[0])
	}
	for i := 1; i < numSlots / hidDim; i *= 2 {
		tmp := partSum[0].CopyNew()
		// eval.Rotate(partSum[0], i, tmp)
		rotateAnyStep(eval, partSum[0], i, tmp)
		eval.Add(partSum[0], tmp, partSum[0])
	}
	y, err = eval.MulNew(partSum[0], mask[0])
	eval.Rescale(y, y)
	if err != nil {
		panic(err)
	}
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), s.Level(), y.Level())

	return y
}

func (llama *LlamaInference) AttnVMsg(s []complex128) (y []complex128) {
	vCache := llama.cacheMsg["v"]
	hidDim := llama.size.hidDim
	numHeads := llama.size.numHeads

	for j := 0; j < hidDim / numHeads; j++ {
		for i := 0; i < numHeads; i++ {
			sum := complex128(0.0)
			for k := 0; k < len(s) / numHeads; k++ {
				sum += s[i + k * numHeads] * vCache[k][i + j * numHeads]
			}
			y = append(y, sum)
		}
	}

	return y
}

func (llama *LlamaInference) Out(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	if _, ok := llama.w["out"]; !ok {
		panic("Executing Out, but w_out not initialized!")
	}

	fmt.Print("Computing Out...\n")
	start := time.Now()
	y = llama.Linear(x, "out", 0)
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), x.Level(), y.Level())

	return y
}

func (llama *LlamaInference) UpGate(x *rlwe.Ciphertext) (up *rlwe.Ciphertext, gate *rlwe.Ciphertext) {
	if _, ok := llama.w["up"]; !ok {
		panic("Executing Up and Gate, but w_up not initialized!")
	} else if _, ok := llama.w["gate"]; !ok {
		panic("Executing Up and Gate, but w_gate not initialized!")
	}

	fmt.Print("Computing Up/Gate...\n")
	start := time.Now()
	up = llama.Linear(x, "up", 1)
	gate = llama.Linear(x, "gate", 1)
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), x.Level(), up.Level())

	return up, gate
}

func (llama *LlamaInference) Down(x *rlwe.Ciphertext) (y *rlwe.Ciphertext) {
	if _, ok := llama.w["down"]; !ok {
		panic("Executing Down, but w_down not initialized!")
	}

	fmt.Print("Computing Down...\n")
	start := time.Now()
	y = llama.Linear(x, "down", -1)
	elapsed := time.Since(start)
	fmt.Printf("Consumed %f seconds with input level %d and output level %d\n", elapsed.Seconds(), x.Level(), y.Level())

	return y
}
