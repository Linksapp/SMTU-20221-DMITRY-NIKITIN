import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5)

def neuron_w(input_count):
        weights = np.zeros(input_count+1)
        for i in range(1, (input_count+1)):
            weights[i] = np.random.uniform(-1.0, 1.0)
        return weights

n_w = [neuron_w(2), neuron_w(2), neuron_w(2), neuron_w(2), neuron_w(1), neuron_w(2)]

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

x_train = [np.array(i) for i in list(zip(x0, x1, x2, x3, x4))]
# y_train = [np.array(i) for i in ]

class Perceptron:
    LEARNING_RATE = 0.1

    def __init__(self, w: list[int | float], x_train: list[int], y_train: list[int]) -> None:
        self.w = w
        self.x_train = x_train
        self.y_train = y_train
        self.n_y = [0, 0, 0, 0]
        self.n_error = [0, 0, 0, 0]
        self.new_weights = []
        self.old_weights = []
        self.y_out = [0 for i in y_train]
        self.index_list = [i for i in range(len(self.x_train))]
    
    def show_learning(self, w: list[int | float]) -> None:
        if len(w) == 3:
            print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])
        else:
            print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1])

    def forward_pass(self, x): 
        self.n_y[0] = np.tanh(np.dot(self.w[0], np.array([x[0], x[1], x[3]]))) 
        self.n1_inputs = np.array([x[0], self.n_y[0], x[4]]) 
        self.n_y[1] = np.tanh(np.dot(self.w[1], self.n1_inputs)) 
        self.n2_inputs = np.array([x[0], self.n_y[1]])
        self.n_y[2] = np.tanh(np.dot(self.w[2], self.n2_inputs))
        self.n3_inputs = np.array([x[0], self.n_y[2], x[2]])
        self.argument = np.dot(self.w[3], self.n3_inputs)
        self.n_y[3] = 1.0 / (1.0 + np.exp(-self.argument)) 

    def backward_pass(self, y_truth): 
        self.error_prime = -(y_truth - self.n_y[3]) # Вычисление ошибки: e = y - d
        self.derivative = self.n_y[3] * (1.0 - self.n_y[3]) 
        self.n_error[3] = self.error_prime * self.derivative 
        self.derivative = 1.0 - self.n_y[0]**2 
        self.n_error[0] = self.w[2][1] * self.n_error[2] * self.derivative 
        self.derivative = 1.0 - self.n_y[1]**2  
        self.n_error[1] = self.w[2][2] * self.n_error[2] * self.derivative 

    def adjust_weights(self, x): # определяем функцию, которая корректирует веса
        self.w[0] -= (x * Perceptron.LEARNING_RATE * self.n_error[0])  # задаём первый вес путём вычитания из массива n_w[0] массива x, умноженного на произведение LEARNING_RATE и n_error[0]
        self.w[1] -= (x * Perceptron.LEARNING_RATE * self.n_error[1])  # задаём второй вес путём вычитания из массива n_w[1] массива x, умноженного на произведение LEARNING_RATE и n_error[1]
        self.n2_inputs = np.array([1.0, self.n_y[0], self.n_y[1]]) # создаём массив входных данных
        self.w[2] -= (self.n2_inputs * Perceptron.LEARNING_RATE * self.n_error[2]) # задаем третий вес путём вычитания из массива n_w[2] массива n2_inputs, умноженного на произведение LEARNING_RATE и n_error[2]

        
    def learning(self) -> None:
        all_correct = False  # задаём предполагаемую правильность выставленных весов
        while not all_correct: # Обучаем сеть поа веса не будут подобраны правильно
            all_correct = True  # если веса подобраны правильно, то цикл обучения закончится
            np.random.shuffle(self.index_list) # Задаём случайный порядок
            for i in self.index_list: # Тренеруем сеть на всех примерах
                self.forward_pass(self.x_train[i])    # подсчитываем выходные значения всех нейронов
                self.backward_pass(self.y_train[i])   # вычисляем погрешность
                self.adjust_weights(self.x_train[i])  # корректируем веса
                self.show_learning() # Выводим обновленные веса
            for i in range(len(self.x_train)): # Проверяем на сходимость
                self.forward_pass(self.x_train[i]) # подсчитываем выходные значения всех нейронов на новых весах
                print(f'Выходное значение: {self.n_y[i]} => {self.y_train[i]}')
                if(((self.y_train[i] < 0.5) and (self.n_y[2] >= 0.5))  # проверяем на несходимость
                        or ((self.y_train[i] >= 0.5) and (self.n_y[2] < 0.5))):
                    all_correct = False # не заканчиваем цикл обучения

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
network = Perceptron(n_w[0], x_train, y4)
network.learning()
# show_progress(and_perc.old_weights, and_perc.new_weights, 0, 0)
# print('------------')
# xor1_perc = Perceptron(w2_1, list(zip(x0, and_perc.y_out, x4)), y2_1)
# xor1_perc.learning()
# print('------------')
# xor2_perc = Perceptron(w2_2, list(zip(x0, and_perc.y_out, x4)), y2_2)
# xor2_perc.learning()
# print('------------')
# xor3_perc = Perceptron(w2_3, list(zip(x0, xor1_perc.y_out, xor2_perc.y_out)), y2_3)
# xor3_perc.learning()
# show_progress(xor3_perc.old_weights, xor3_perc.new_weights, 0, 1)
# print('------------')
# inv_perc = Perceptron(w3, list(zip(x0, xor3_perc.y_out)), y3)
# inv_perc.learning()
# show_progress(inv_perc.old_weights, inv_perc.new_weights, 1, 0)
# print('------------')
# sum_perc = Perceptron(w4, list(zip(x0, inv_perc.y_out, x2)), y4)
# sum_perc.learning()
# show_progress(sum_perc.old_weights, sum_perc.new_weights, 1, 1)

# plt.show()

# fig, axs = plt.subplots(2, 2, figsize=(12, 7))

# def show_final(new_weights: list[int | float], x_train: list[int], ind_1: int, ind_2: int):
#     axs[ind_1, ind_2].axis(xmin=-2, xmax=2, ymin=-2, ymax=2)
#     axs[ind_1, ind_2].set_xlabel('x1')
#     axs[ind_1, ind_2].set_ylabel('x2')
#     fig.suptitle('Конечные значения весов после обучения')

#     if len(new_weights[0]) == 3:
#         axs[ind_1, ind_2].plot([-new_weights[-1][1]/new_weights[-1][2] * -2 - new_weights[-1][0] / new_weights[-1][2], 
#             -new_weights[-1][1]/new_weights[-1][2] * 2 - new_weights[-1][0] / new_weights[-1][2]], [-2, 2])
        
#     for i in range(4):
#         z = 0
#         for j in range(len(new_weights[-1])):
#             z += x_train[i][j] * new_weights[-1][j]
#         print(x_train[i], new_weights[-1], z)
#         if z < 0:
#             axs[ind_1, ind_2].scatter(x_train[i][2], x_train[i][1],  marker='_', c='red')
#         else:
#             axs[ind_1, ind_2].scatter(x_train[i][2], x_train[i][1],  marker='+', c='green')
    
    
    

# show_final(and_perc.new_weights, points, 0, 0)
# show_final(xor1_perc.new_weights, points, 0, 1)
# show_final(xor2_perc.new_weights, points, 1, 0)
# show_final(sum_perc.new_weights, points, 1, 1)

# plt.show()

# for i in zip(and_perc.y_out, xor3_perc.y_out, inv_perc.y_out, sum_perc.y_out):
#     print(i)