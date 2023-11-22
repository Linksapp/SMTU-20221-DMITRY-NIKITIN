import random
import matplotlib.pyplot as plt

random.seed(5)
w1 = [0.5, 0.5, 0.5]
w2_1 = [0.5, 0.5, 0.5]
w2_2 = [0.5, 0.5, 0.5]
w2_3 = [0.5, -0.5, 0.5]
w3 = [0.5, 0.5]
w4 = [-0.5, 0.5, 0.5]

x0 = [1 for i in range(16)]
x1 = [1 if i // 8 == 1 else -1 for i in range(16)]
x2 = [1 if i // 4 % 2 != 0 else -1 for i in range(16)]
x3 = [1 if i // 2 % 2 != 0 else -1 for i in range(16)]
x4 = [1 if i % 2 != 0 else -1 for i in range(16)]

y1 = [1 if (x1[i] == 1 and x3[i] == 1) else -1 for i in range(16)]
y2_1 = [1 if (y1[i] == 1 and x4[i] == -1) else -1 for i in range(16)]
y2_2 = [1 if (y1[i] == -1 and x4[i] == 1) else -1 for i in range(16)]
y2_3 = [1 if y2_1[i] == 1 or y2_2[i] == 1 else -1 for i in range(16)]
y3 = [1 if y2_3[i] == -1 else -1 for i in range(16)]
y4 = [1 if y3[i] == 1 or x2[i] == 1 else -1 for i in range(16)]

class Perceptron:
    LEARNING_RATE = 0.1

    def __init__(self, w: list[int | float], x_train: list[int], y_train: list[int]) -> None:
        self.w = w
        self.x_train = x_train
        self.y_train = y_train
        self.new_weights = []
        self.old_weights = []
        self.y_out = [0 for i in y_train]
        self.index_list = [i for i in range(len(self.x_train))]

    def show_learning(self, w: list[int | float]) -> None:
        if len(w) == 3:
            print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])
        else:
            print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1])

    def compute_output(self, w: list[int | float], x: list[int]) -> int:
        z = 0.0
        
        for i in range(len(w)):
            z += x[i] * w[i] # Вычисление суммы взвешенных входов

        
        if z < 0: # Применение знаковой функции
            return -1
        else:
            return 1
        
    def learning(self) -> None:
        all_correct = False
        random.shuffle(self.index_list)
        while not all_correct:
            all_correct = True
            for i in self.index_list:
                self.x = self.x_train[i]
                self.y = self.y_train[i]
                self.p_out = self.compute_output(self.w, self.x)

                if self.y != self.p_out:
                    self.old_weights.append(self.w.copy())
                    for j in range(0, len(self.w)):
                        self.w[j] += (self.y * Perceptron.LEARNING_RATE * self.x[j])
                    all_correct = False
                    self.new_weights.append(self.w.copy())
                    self.show_learning(self.w)
                self.y_out[i] = self.compute_output(self.w, self.x)

fig, axs = plt.subplots(2, 2, figsize=(12, 7))


def show_progress(old_weights: list[int | float], new_weights: list[int | float], ind_1: int, ind_2: int) -> None:
    axs[ind_1, ind_2].axis(xmin=-2, xmax=2, ymin=-2, ymax=2)
    axs[ind_1, ind_2].set_xlabel('x1')
    axs[ind_1, ind_2].set_ylabel('x2')
    fig.suptitle('Процесс обучения')

    if len(old_weights[0]) == 3:
        for i in range(len(new_weights)):
            axs[ind_1, ind_2].plot([-2, 2], [-old_weights[i][1]/old_weights[i][2] * -2 - old_weights[i][0]/old_weights[i][2], 
                                -new_weights[i][1] / new_weights[i][2] * 2 - new_weights[i][0] / new_weights[i][2]], 
                                '--' if i == len(new_weights)-1 else '-')
    else: 
        for i in range(len(new_weights)):
            axs[ind_1, ind_2].plot([-2, 2], [-old_weights[i][0]/old_weights[i][1], 
                                -new_weights[i][0]/new_weights[i][1]], 
                                '--' if i == len(new_weights)-1 else '-')    
        
points = [(1, 1 if i >= 2 else -1, 1 if i % 2 != 0 else -1) for i in range(4)]
and_perc = Perceptron(w1, list(zip(x0, x1, x3)), y1)
and_perc.learning()
show_progress(and_perc.old_weights, and_perc.new_weights, 0, 0)
print('------------')
xor1_perc = Perceptron(w2_1, list(zip(x0, and_perc.y_out, x4)), y2_1)
xor1_perc.learning()
print('------------')
xor2_perc = Perceptron(w2_2, list(zip(x0, and_perc.y_out, x4)), y2_2)
xor2_perc.learning()
print('------------')
xor3_perc = Perceptron(w2_3, list(zip(x0, xor1_perc.y_out, xor2_perc.y_out)), y2_3)
xor3_perc.learning()
show_progress(xor3_perc.old_weights, xor3_perc.new_weights, 0, 1)
print('------------')
inv_perc = Perceptron(w3, list(zip(x0, xor3_perc.y_out)), y3)
inv_perc.learning()
show_progress(inv_perc.old_weights, inv_perc.new_weights, 1, 0)
print('------------')
sum_perc = Perceptron(w4, list(zip(x0, inv_perc.y_out, x2)), y4)
sum_perc.learning()
show_progress(sum_perc.old_weights, sum_perc.new_weights, 1, 1)

plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 7))

def show_final(new_weights: list[int | float], x_train: list[int], ind_1: int, ind_2: int):
    axs[ind_1, ind_2].axis(xmin=-2, xmax=2, ymin=-2, ymax=2)
    axs[ind_1, ind_2].set_xlabel('x1')
    axs[ind_1, ind_2].set_ylabel('x2')
    fig.suptitle('Конечные значения весов после обучения')

    if len(new_weights[0]) == 3:
        axs[ind_1, ind_2].plot([-new_weights[-1][1]/new_weights[-1][2] * -2 - new_weights[-1][0] / new_weights[-1][2], 
            -new_weights[-1][1]/new_weights[-1][2] * 2 - new_weights[-1][0] / new_weights[-1][2]], [-2, 2])
        
    for i in range(4):
        z = 0
        for j in range(len(new_weights[-1])):
            z += x_train[i][j] * new_weights[-1][j]
        print(x_train[i], new_weights[-1], z)
        if z < 0:
            axs[ind_1, ind_2].scatter(x_train[i][2], x_train[i][1],  marker='_', c='red')
        else:
            axs[ind_1, ind_2].scatter(x_train[i][2], x_train[i][1],  marker='+', c='green')
    
    
    

show_final(and_perc.new_weights, points, 0, 0)
show_final(xor1_perc.new_weights, points, 0, 1)
show_final(xor2_perc.new_weights, points, 1, 0)
show_final(sum_perc.new_weights, points, 1, 1)

plt.show()

for i in zip(and_perc.y_out, xor3_perc.y_out, inv_perc.y_out, sum_perc.y_out):
    print(i)