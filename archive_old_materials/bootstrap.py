import numpy as np
import pandas as pd
import argparse
import copy
import time

boot_lat = 24
max_level = 13

class Layer:
    def __init__(self, name, lat):
        self.name = name
        self.lat = lat
        self.depth = 0
        if type(lat[0]) != list:
            for l in lat:
                if l == np.inf:
                    self.depth += 1
            self.compose = False
        else:
            self.compose = True
        self.min_lat = [np.inf] * (
            max_level + 1
        )  # minimum latency before this layer when input level = i
        self.route = [
            [] for i in range(max_level + 1)
        ]  # the level assignment including the input of this layer when output level = i


class Bootplacer:
    def __init__(self, name, choose_min=False):
        self.layers = []
        self.name = name
        self.choose_min = choose_min

    def add_layer(self, name, lat):
        new_layer = Layer(name, lat)
        self.layers.append(new_layer)

    def solve(self, prune):
        output = [[np.inf for _ in range(max_level + 1)] for _ in range(max_level + 1)]
        for i in range(max_level + 1):  # search every source
            print(f"Search using the origin {i}...")

            self.layers[0].min_lat = [0 for _ in self.layers[0].min_lat]
            for j in range(len(self.layers) - 1):
                if self.layers[j].compose == False:
                    if j == 0:
                        for k in range(max_level - self.layers[0].depth + 1):
                            self.layers[1].min_lat[k] = self.layers[0].lat[
                                k + self.layers[0].depth
                            ]
                            if i < k + self.layers[0].depth:
                                self.layers[1].min_lat[k] += boot_lat
                            self.layers[0].route[k] = [(self.layers[0].name, i)]
                    else:
                        for k in range(max_level - self.layers[j].depth + 1):
                            if prune:
                                from_higher = (
                                    self.layers[j].min_lat[k + self.layers[j].depth]
                                    + self.layers[j].lat[k + self.layers[j].depth]
                                )
                                from_lower = (
                                    self.layers[j].min_lat[0]
                                    + self.layers[j].lat[k + self.layers[j].depth]
                                    + boot_lat
                                )
                                if from_higher < from_lower:
                                    self.layers[j + 1].min_lat[k] = from_higher
                                    self.layers[j].route[k] = copy.deepcopy(
                                        self.layers[j - 1].route[
                                            k + self.layers[j].depth
                                        ]
                                    )
                                    self.layers[j].route[k].append(
                                        (self.layers[j].name, k + self.layers[j].depth)
                                    )
                                else:
                                    self.layers[j + 1].min_lat[k] = from_lower
                                    self.layers[j].route[k] = copy.deepcopy(
                                        self.layers[j - 1].route[0]
                                    )
                                    self.layers[j].route[k].append(
                                        (self.layers[j].name, 0)
                                    )
                            else:
                                for l in range(max_level + 1):
                                    lat = (
                                        self.layers[j].min_lat[l]
                                        + self.layers[j].lat[k + self.layers[j].depth]
                                    )
                                    if l < k + self.layers[j].depth:
                                        lat += boot_lat
                                    if lat < self.layers[j + 1].min_lat[k]:
                                        self.layers[j + 1].min_lat[k] = lat
                                        self.layers[j].route[k] = copy.deepcopy(
                                            self.layers[j - 1].route[l]
                                        )
                                        self.layers[j].route[k].append(
                                            (self.layers[j].name, l)
                                        )
                else:
                    if j == 0:
                        for k in range(max_level + 1):
                            self.layers[1].min_lat[k] = self.layers[0].lat[i][k]
                            self.layers[0].route[k] = [(self.layers[0].name, i)]
                    else:
                        for k in range(max_level + 1):
                            for l in range(max_level + 1):
                                if (
                                    self.layers[j + 1].min_lat[l]
                                    > self.layers[j].min_lat[k]
                                    + self.layers[j].lat[k][l]
                                ):
                                    self.layers[j + 1].min_lat[l] = (
                                        self.layers[j].min_lat[k]
                                        + self.layers[j].lat[k][l]
                                    )
                                    self.layers[j].route[l] = copy.deepcopy(
                                        self.layers[j - 1].route[k]
                                    )
                                    self.layers[j].route[l].append(
                                        (self.layers[j].name, k)
                                    )

            min_lat = np.inf
            for j in range(max_level + 1):
                output[i][j] = self.layers[-1].min_lat[j]
                if min_lat > output[i][j]:
                    min_lat = output[i][j]
                if self.layers[-1].min_lat[j] != np.inf:
                    print(
                        f"Output of {self.name} with level {j} requires {self.layers[-1].min_lat[j]:.2f} seconds, the route being: "
                    )
                    print(self.layers[-2].route[j])

        if self.choose_min:
            print(f"The minimal end-to-end latency is {min_lat}.")

        return output


def add_shortcut(data):
    for i in range(len(data)):
        for j in range(i):
            data[j][i] += boot_lat

    return data


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
        # import pdb; pdb.set_trace()

    if "RoPE" in result_dict:
        result_dict["Cache"] = [
            result_dict["RoPE"][i] + result_dict["Cache"][i] for i in range(len(result_dict["RoPE"]))
        ]
        del result_dict["RoPE"]
    return result_dict


# def solve_shortcut_softmax(data, prune):
#     placer = Bootplacer("Shortcut_softmax")
#     placer.add_layer(f"GoldIter", data["Softmax"])
#     placer.add_layer("mult", data["CtMult"])
#     placer.add_layer("last_layer", [0] * (max_level + 1))

#     return placer.solve(prune)


# def solve_softmax(data, prune):
#     placer = Bootplacer("Softmax")
#     lat_shortcut = add_shortcut(solve_shortcut_softmax(data, prune))
#     for i in range(8):
#         placer.add_layer(f"mult_{i}", data["CtMult"])
#     placer.add_layer("shortcut", lat_shortcut)
#     placer.add_layer("last_layer", [0] * (max_level + 1))

#     return placer.solve(prune)


# def solve_shortcut_norm(data, prune):
#     placer = Bootplacer("Shortcut_norm")
#     # lat_newton = add_shortcut(solve_iter_norm(data))
#     # placer.add_layer("var", data["Var"])
#     # placer.add_layer("mult", data["CtMult"])
#     placer.add_layer("Newton", data["SqrtNt"])
#     placer.add_layer("GoldIter", data["SqrtGold"])
#     placer.add_layer("mult", data["CtMult"])
#     placer.add_layer("last_layer", [0] * (max_level + 1))

#     return placer.solve(prune)


# def solve_norm(data, prune):
#     placer = Bootplacer("Norm")
#     lat_shortcut = add_shortcut(solve_shortcut_norm(data, prune))
#     placer.add_layer("mult", data["CtMult"])
#     placer.add_layer("shortcut", lat_shortcut)
#     placer.add_layer("last_layer", [0] * (max_level + 1))

#     return placer.solve(prune)


def solve_qk(data, prune):  # do not include QKV
    print("Solving qk...\n")
    placer = Bootplacer("QK")
    lat_softmax = data["Softmax"]
    placer.add_layer("Cache", data["Cache"])  # includes RoPE
    placer.add_layer("QK_T", data["QK_T"])
    placer.add_layer("Softmax", lat_softmax)
    placer.add_layer("last_layer", [0] * (max_level + 1))
    return placer.solve(prune)


def solve_MHA(data, prune):
    print("Solving MHA...\n")
    placer = Bootplacer("MHA")
    lat_qk = add_shortcut(solve_qk(data, prune))
    lat_norm = data["Norm"]
    placer.add_layer("Norm", lat_norm)
    placer.add_layer("QKV", data["QKV"])
    placer.add_layer("RoPE_to_AttnV", lat_qk)
    placer.add_layer("AttnV", data["AttnV"])
    placer.add_layer("O", data["Out"])
    placer.add_layer("last_layer", [0] * (max_level + 1))

    return placer.solve(prune)


def solve_shortcut_silu(data, prune):
    placer = Bootplacer("SiLU")
    placer.add_layer("SiLU", data["SiLU"])
    placer.add_layer("last_layer", [0] * (max_level + 1))

    return placer.solve(prune)


def solve_FFN(data, prune):
    print("Solving FFN...\n")
    placer = Bootplacer("FFN")
    lat_norm = data["Norm"]
    lat_silu = add_shortcut(solve_shortcut_silu(data, prune))
    placer.add_layer("Norm", lat_norm)
    placer.add_layer("UpGate", [2 * t for t in data["UpGate"]])
    placer.add_layer("SiLU", lat_silu)
    placer.add_layer("elem_mult", [0] * (max_level + 1))
    placer.add_layer("Down", data["Down"])
    placer.add_layer("last_layer", [0] * (max_level + 1))

    return placer.solve(prune)


def solve_decoder(data, prune):
    print("Solving decoder...\n")
    placer = Bootplacer("Decoder")
    lat_MHA = add_shortcut(solve_MHA(data, prune))
    lat_FFL = add_shortcut(solve_FFN(data, prune))
    if prune:
        placer.add_layer("MHA", lat_MHA)
        placer.add_layer("FFL", lat_FFL)
    else:
        for i in range(32):
            placer.add_layer("MHA_i", lat_MHA)
            placer.add_layer("FFL_i", lat_FFL)
    placer.add_layer("last_layer", [0] * (max_level + 1))

    return placer.solve(prune)


def solve_model(data, prune):
    print("Solving Model...\n")
    placer = Bootplacer("Model", True)
    lat_decoder = solve_decoder(data, prune)
    if prune:
        for i in range(32):
            placer.add_layer(f"decoder_{i}", lat_decoder)
    else:
        placer.add_layer(f"decoder", lat_decoder)
    placer.add_layer("last_layer", [0] * (max_level + 1))

    return placer.solve(prune)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="./new_data.csv")
    parser.add_argument("--prune", type=int, default=1)
    args = parser.parse_args()
    data = read_data(args.file)
    solve_model(data, args.prune)
    return


if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    print("Latency:", (end - begin) * 1000, "ms")
