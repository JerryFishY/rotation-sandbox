import numpy as np
import pandas as pd
from collections import defaultdict

boot_lat = 24
max_level = 13

def read_data(file_name):
    df = pd.read_csv(file_name, sep="\s+", dtype={0: str}, header=None)
    result_dict = {}
    for _, row in df.iterrows():
        values = row.values.tolist()
        name = values[0]
        values[0] = np.inf
        for i, value in enumerate(values):
            if value == 0:
                values[i] = np.inf
            if type(value) == str:
                values[i] = float(value.replace(",", ""))
        result_dict[name] = values
        
    if "RoPE" in result_dict:
        result_dict["Cache"] = [
            result_dict["RoPE"][i] + result_dict["Cache"][i] for i in range(len(result_dict["RoPE"]))
        ]
        del result_dict["RoPE"]
    return result_dict

if __name__ == "__main__":
    data = read_data('../benchmarks_reference/new_data.csv')
    
    # Calculate pure operations latency
    ops_sum = 0
    print("=== Pure Operation Times ===")
    for k, v in data.items():
        if k == 'UpGate':
            # UpGate is added as 2 * t
            min_op = min([val for val in v if val != np.inf]) * 2
        elif k == 'RoPE':
            continue # already in cache
        else:
            min_op = min([val for val in v if val != np.inf])
        print(f"{k}: {min_op:.2f}s")
        ops_sum += min_op
        
    print(f"\nSum of minimal pure operations: {ops_sum:.2f}s")
    
    # From python output: 160.60 seconds per layer
    # So boots time = 160.60 - ops_sum
    tot_layer = 5139.367 / 32
    print(f"Total time per layer from bootstrap.py: {tot_layer:.2f}s")
    boots_time = tot_layer - ops_sum
    print(f"Total time spent on bootstrappings per layer: {boots_time:.2f}s")
    print(f"Total boots count per layer: {boots_time / boot_lat:.2f} boots")
