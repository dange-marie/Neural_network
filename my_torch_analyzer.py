#!/usr/bin/env python3
from my_mlp import MLP
from params import Params
import sys

def get_output(output):
    if output[0][0] > 0.5:
        return "Check"
    if output[1][0] > 0.5:
        return "Nothing"
    if output[2][0] > 0.5:
        return "Checkmate"
    if output[3][0] > 0.5:
        return "Stalemate"

def main():
    try:
        param = Params(sys.argv)
        brain = MLP(param.load_file)
        if param.train:
            brain.load_training_data_set(param.file)
            if param.save:
                brain.train(param.save_file)
            else:
                brain.train(param.load_file)
        if param.predict:
            brain.load_training_data_set(param.file)
            for i in brain.content:
                if len(i.split(' ')) < 2:
                    continue
                inputs, output = brain.fen_to_input(i)
                res = brain.predict(inputs)
                print(res)
    except BrokenPipeError:
        pass
    except KeyboardInterrupt:
        if param.save:
            brain.save(param.save_file)
        else:
            brain.save(param.load_file)
        exit(0)

if __name__ == "__main__":
    main()
