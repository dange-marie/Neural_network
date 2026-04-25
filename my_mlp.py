#!/usr/bin/env python3
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import time

pawn_scores = {
    "K": -4,
    "R": -3,
    "B": -3,
    "N": -2,
    "Q": -2,
    "P": -1,
    "k": 4,
    "r": 3,
    "b": 3,
    "n": 2,
    "q": 2,
    "p": 1
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.where(x > 0, x, 0.01 * x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def softmax_derivative(x):
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

class MLP:
    def __init__(self, mlp_config_file: str, verbose=False):
        with open(mlp_config_file) as json_file:
            data = json.load(json_file)
        self.nb_inputs = data["nb_inputs"]
        self.nb_layers = data["nb_layers"]
        self.params = {}
        for key, value in data["params"].items():
            if key[0] != "F":
                self.params[key] = np.array(value)
            else:
                self.params[key] = value
            if verbose:
                print("Parameters ", key, " ", self.params[key].shape)
        self.learning_rate = data["learning_rate"]

    def save(self, filename):
        save = {}
        for key, value in self.params.items():
            if key[0] != "F":
                save[key] = value.tolist()
            else:
                save[key] = value
        dict = {
            "nb_inputs": self.nb_inputs,
            "nb_layers": self.nb_layers,
            "learning_rate": self.learning_rate,
            "params": save,
        }
        with open(filename, "w") as file:
            json.dump(dict, file, indent=4)

    def load_training_data_set(self, filename):
        with open(filename) as f:
            self.content = f.read().split("\n")

    def fen_to_input(self, fen_string: str):
        res = [[0] for i in range(78)]
        output = [[0] for i in range(4)]
        array = fen_string.split(' ')
        board = array[0].split('/')
        row = 0
        for i in board:
            if row > 8:
                exit(84)
            line = [k for k in i]
            pion = 0
            for elem in line:
                if pion > 8:
                    exit(84)
                if elem.isnumeric():
                    pion += int(elem)
                else:
                    res[(row * 8) + pion][0] = pawn_scores[elem]
                    pion += 1
            row += 1
        if len(array) > 1:
            if array[1] == "b":
                res[64][0] = 0
            else:
                res[64][0] = 1

        if len(array) > 2:
            roque = array[2]
            res[65][0] = 1 if 'K' in roque else 0
            res[66][0] = 1 if 'Q' in roque else 0
            res[67][0] = 1 if 'k' in roque else 0
            res[68][0] = 1 if 'q' in roque else 0

        if len(array) > 3 and array[3] != '-':
            en_passant_col = array[3][0]
            column_index = ord(en_passant_col) - ord('a')
            res[69 + column_index][0] = 1

        if len(array) > 6:
            if array[6] == "Check":
                output[0][0] = 1
            if array[6] == "Nothing":
                output[1][0] = 1
            if array[6] == "Checkmate":
                output[2][0] = 1
            if array[6] == "Stalemate":
                output[3][0] = 1
        
        # if len(array) > 7:
        #     if array[7] == "White":
        #         output[4][0] = 1

        return np.array(res), np.array(output)

    def compute_cost(self, output, target):
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        return (1 / len(target) * np.sum(-target * np.log(output) - (1 - target) * np.log(1 - output)))

    def cross_entropy(self, output, target):
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        return -np.sum(target * np.log(output)) / target.shape[1]

    def compute_layer(self, inputs, weights, bias, activation="sigmoid"):
        matrice = weights.dot(inputs) + bias
        if activation == "sigmoid":
            return sigmoid(matrice)
        elif activation == "relu":
            return relu(matrice)
        elif activation == "tanh":
            return tanh(matrice)
        elif activation == "softmax":
            return softmax(matrice)
        else:
            return matrice

    def forward_propagation(self, inputs, verbose=False):
        activation = {"A0": inputs}
        for i in range(1, self.nb_layers + 1):
            activation[f"A{i}"] = self.compute_layer(activation[f"A{i-1}"], self.params[f"W{i}"], self.params[f"b{i}"], activation=self.params[f"F{i}"])
        if verbose:
            for key, value in activation.items():
                print(key, ": ", value.shape)
        return activation

    def derivative(self, output, activation_type):
        if activation_type == "sigmoid":
            return sigmoid_derivative(output)
        elif activation_type == "relu":
            return relu_derivative(output)
        elif activation_type == "tanh":
            return tanh_derivative(output)
        elif activation_type == "softmax":
            return softmax_derivative(output)
        else:
            return output

    def backward_propagation(self, target, output, verbose=False):
        m = target.shape[1]
        dz = output[f'A{self.nb_layers}'] - target
        gradients = {}
        for i in reversed(range(1, self.nb_layers + 1)):
            gradients[f"dw{i}"] = 1 / m * np.dot(dz, output[f"A{i-1}"].T)
            gradients[f"db{i}"] = 1 / m * np.sum(dz, axis=1, keepdims=True)
            if i > 1:
                dz = np.dot(self.params[f"W{i}"].T, dz) * self.derivative(output[f"A{i-1}"], self.params[f"F{i-1}"])
        if verbose:
            for key, value in gradients.items():
                print(key, ": ", value.shape)
        return gradients

    def update(self, gradients):
        for i in range(1, self.nb_layers + 1):
            self.params[f"W{i}"] -= self.learning_rate * gradients[f"dw{i}"]
            self.params[f"b{i}"] -= self.learning_rate * gradients[f"db{i}"]

    def predict(self, inputs, verbose=False):
        forward = self.forward_propagation(inputs)
        if verbose:
            print(forward[f"A{self.nb_layers}"])
        output = forward[f"A{self.nb_layers}"]
        color = ""
        if output.shape[1] == 5:
            if output[4][0] > 0.5:
                color = "White"
            else:
                color = "Black"
        if output[0][0] > 0.5:
            return "Check"+color
        if output[1][0] > 0.5:
            return "Nothing"+color
        if output[2][0] > 0.5:
            return "Checkmate"+color
        if output[3][0] > 0.5:
            return "Stalemate"+color
        return "Nothing"

    def sample_generator(self):
        for i in range(0, len(self.content), 100000):
            yield self.content[i:i + 100000]

    def train(self, save_file):
        self.percent = 0
        max_iterations = 4
        cost = 0
        old_cost = 0
        lr = self.learning_rate
        np.random.shuffle(self.content)
        start_time = time.time()
        epochs = []
        epoch_costs = []
        while True:
            self.percent = 0
            cost = 0
            sample = 1
            for content in self.sample_generator():
                streak = 0
                for elem in content:
                    print(f"Epoch {4 - max_iterations}: => Processing Line \033[0;34m{streak} of samples {sample}\033[0m 🧠")
                    inputs, target = self.fen_to_input(elem)
                    ac = self.forward_propagation(inputs)
                    gradients = self.backward_propagation(target, ac)
                    self.update(gradients)
                    cost += self.cross_entropy(ac[f"A{self.nb_layers}"], target)
                    streak += 1
                print(f"\033[0;32mEnd of sample {sample}\033[0m")
                sample += 1
            cost /= len(self.content)
            epochs.append(5 - max_iterations)
            epoch_costs.append(cost)
            self.save(save_file)
            print(f"\033[0;32mComplete {4 - max_iterations} epoch 🮱\033[0m")
            max_iterations -= 1
            if max_iterations == 0:
                break
            if abs(cost - old_cost) < 0.0001:
                break
            print(10 - max_iterations)
            old_cost = cost
            self.learning_rate *= 0.95
        self.learning_rate = lr
        self.save(save_file)
        end_time = time.time()
        total_time = end_time - start_time
        print("Training complete on the epoch ", 4 - max_iterations, " with error differential ", self.percent)   
        print(f"\033[0;36mTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)\033[0m")

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, epoch_costs, marker='o', label="Cost per Epoch")
        plt.title("Training Progress: Cost vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Cost (Loss)")
        plt.grid(True)
        plt.legend()
        plt.show()
