# Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language Model with KV Cache

We present our implementation of Cachemir here.

To verify the availability of our end-to-end implementation with optimal level assigned to each layer very quickly, run:

```bash
go run .
```

Also, one can run the full version, which gives our main results:

```bash
go run . -logN=16 -hidDim=4096 -expDim=16384 -seqLen=512
```

To verify the availability of our bootstrapping placement algorithm, run: 

```bash
python3 bootstrap.py --prune=1
```

To reproduce our results in a more fine-grained manner, run the bash script, which make take some time. The script first evaluate all the modules and store the results into a .csv file, and then run the bootstrapping placement algorithm to compute the minimized overall latency:

```bash
bash run_test.sh
```

For easy comparison with prior works that are not implemented using Lattigo, we also provide convenient evaluation of homomorphic operations:

```bash
go run . -test=Ops -level=[level] -logN=16
```

One can replace level with any integer from 1 to 16.
