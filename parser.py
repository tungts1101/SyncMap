import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

class Parser:
    def __init__(self):
        self.run()


    def run(self):
        self.train_data = np.array(pd.read_csv("data/mnist_train.csv"))
        self.test_data = np.array(pd.read_csv("data/mnist_test.csv"))

        # image = train_data[50, 1:]
        # image = np.expand_dims(image, axis=0).reshape(28, 28).astype("uint8")
        # plt.imshow(image)
        # plt.show()

    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data
    
    def get_train_data_for_one_image(self, data):
        states = []
        for _ in range(100):
            state = [0 for _ in range(784 * 255)]
            for _ in range(30):
                i = np.random.randint(1, 785)
                state[i] = data[i]
            states.append(state)
        return states

    def get_sequence(self, sequence_size: int):
        # np.random.shuffle(self.train_data)
        self.true_labels = []
        sequence = []
        for _ in range(sequence_size):
            # sequence.append(self.get_input())
            # self.update()
            i = np.random.randint(self.train_data.shape[0])
            data = self.train_data[i]
            self.true_labels.append(data[0])
            sequence.append(self.get_train_data_for_one_image(data))
        return torch.tensor(np.array(sequence).flatten())
    
    def get_true_labels(self) -> list:
        return self.true_labels
