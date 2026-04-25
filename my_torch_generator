#!/usr/bin/env python3
import sys
import json
import random
import numpy as np

def main():
    if (len(sys.argv) % 2) == 0:
        print("""USAGE
                ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
            DESCRIPTION
                config_file_i Configuration file containing description of a neural network we want
            to generate.
                nb_i Number of neural networks to generate based on the configuration file."""
        )
        sys.exit(84)
    for i in range(1, len(sys.argv), 2):
        config_file = sys.argv[i]
        nb = int(sys.argv[i + 1])
        with open(config_file) as json_file:
            data = json.load(json_file)
        if data["nb_layers"] != len(data["nb_neurons"]):
            print("Error: nb_layers and length of nb_neurons are different.")
            sys.exit(84)
        g = data["nb_neurons"].copy()
        g.insert(0, data["nb_inputs"])
        params = {}
        for c in range(1, len(g)):
            params["W" + str(c)] = (np.random.randn(g[c], g[c-1]) * np.sqrt(2. / g[c-1])).tolist()
            params["b" + str(c)] = (np.random.randn(g[c], 1) * np.sqrt(2. / g[c-1])).tolist()
            params["F" + str(c)] = data["activations"][c-1]
        dict = {
            "nb_inputs": data["nb_inputs"],
            "nb_layers": data["nb_layers"],
            "learning_rate": data["learning_rate"],
            "params": params
        }
        if nb != 1:
            output_file = config_file.split(".")[0] + "_"
            for i in range(nb):
                with open(output_file + str(i+1) + ".nn", "w") as file:
                    json.dump(dict, file, indent=4)
        else:
            output_file = config_file.split(".")[0] + ".nn"
            with open(output_file, "w") as file:
                json.dump(dict, file, indent=4)

if __name__ == "__main__":
    main()
