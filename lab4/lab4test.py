import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(12)

def neuron_w(input_count):
        weights = np.zeros(input_count+1)
        for i in range(1, (input_count+1)):
            weights[i] = np.random.uniform(-1.0, 1.0)
        return weights

n_w = [neuron_w(4) for x in range(9)]

x0 = [1 for i in range(16)]
x1 = [1 if i // 8 == 1 else -1 for i in range(16)]
x2 = [1 if i // 4 % 2 != 0 else -1 for i in range(16)]
x3 = [1 if i // 2 % 2 != 0 else -1 for i in range(16)]
x4 = [1 if i % 2 != 0 else -1 for i in range(16)]

x_train = [np.array(i) for i in list(zip(x0, x1, x2, x3, x4))]
y_train = np.array([1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1]) # 12 - проблема

class Network:
    LEARNING_RATE = 0.1

    def __init__(self, w: list[int | float], x_train: list[int], y_train: list[int]) -> None:
        self.w = w
        self.x_train = x_train
        self.y_train = y_train
        self.n_y = [0 for x in range(len(w))]
        self.n_error = [0 for x in range(len(w))]
        self.index_list = [i for i in range(len(self.x_train))]
        self.epoch = 0
    
    def show_learning(self):
        print(f'epoch - {self.epoch}')
        for i, w in enumerate(n_w):
            print(f'neuron {i}: w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]}, w3 = {w[3]}, w4 = {w[4]}')  
            print('----------------')


    def forward_pass(self, x):
        for i in range(4):
            self.n_y[i] = np.tanh(np.dot(self.w[i], x))
        self.layer2_input = np.array([1.0] + [self.n_y[i] for i in range(4)])
        for i in range(4):
            self.n_y[i+4] = np.tanh(np.dot(self.w[i+4], self.layer2_input))
        self.output_layer_input = np.array([1.0] + [self.n_y[i+4] for i in range(4)])
        self.n_y[8] = 1.0 / (1.0 + np.exp(-np.dot(self.w[8], self.output_layer_input)))

    def backward_pass(self, y):
        self.error_prime = -(y - self.n_y[8])
        self.derivative = self.n_y[8] * (1.0 - self.n_y[8])
        self.n_error[8] = self.error_prime * self.derivative
        for i in range(4):
            self.derivative = 1.0 - self.n_y[i+4]**2
            self.n_error[i+4] = self.w[8][i+1] * self.n_error[8] * self.derivative 
        for i in range(4):
            self.weight_sum = 0
            for j in range(4):
                self.weight_sum += self.n_error[j+4] * self.w[i][j+1]
            self.derivative = 1.0 - self.n_y[i]**2
            self.n_error[i] = self.weight_sum * self.derivative
    
    def adjust_weights(self, x):
        for i in range(4):
            self.w[i] -= (x * Network.LEARNING_RATE * self.n_error[i]) 
            self.w[i+4] -= (self.layer2_input * Network.LEARNING_RATE * self.n_error[i+4])
        self.w[8] -= (self.output_layer_input * Network.LEARNING_RATE * self.n_error[8])
    
    def learning(self):
        all_correct = False  
        while not all_correct and self.epoch != 240: 
            all_correct = True 
            self.epoch += 1
            np.random.shuffle(self.index_list)
            for i in self.index_list: 
                self.forward_pass(self.x_train[i])  
                self.backward_pass(self.y_train[i]) 
                self.adjust_weights(self.x_train[i]) 
                self.show_learning() 
            for i in range(len(self.x_train)): 
                self.forward_pass(self.x_train[i])
                print(f'Выходное значение: {self.n_y[8]} => {self.y_train[i]}')
                if(((self.y_train[i] < 0.5) and (self.n_y[8] >= 0.5))
                        or ((self.y_train[i] >= 0.5) and (self.n_y[8] < 0.5))):
                    all_correct = False 
        
if __name__ == '__main__':
    nn = Network(n_w, x_train, y_train)
    nn.learning()
