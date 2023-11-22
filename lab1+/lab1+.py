import random
import matplotlib.pyplot as plt

class Perceptron:
    LEARNING_RATE = 0.1

    def __init__(self, w: list[int | float], x_train: list[int | float], y_train: list[int]) -> None:
        self.w = w
        self.x_train = x_train
        self.y_train = y_train
        self.new_weights = []

    def show_learning(self, w: list[int | float]) -> None:
        print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2], ', w3 =', '%5.2f' % w[3])

    def compute_output(self, w: list[int | float], x: list[int], y) -> int:
        z = 0.0
        
        for i in range(len(w)):
            z += x[i] * w[i] # Вычисление суммы взвешенных входов

        if z < 0: # Применение знаковой функции
            return -1
        else:
            return 1
        
    def learning(self) -> None:
        all_correct = False
        while not all_correct:
            all_correct = True
            for i in range(len(self.y_train)):
                x = self.x_train[i]
                y = self.y_train[i]
                self.p_out = self.compute_output(self.w, x, y)

                if y != self.p_out:
                    for j in range(0, len(self.w)):
                        self.w[j] += (y * Perceptron.LEARNING_RATE * x[j])
                    all_correct = False
                    self.new_weights.append(self.w.copy())
                    self.show_learning(self.w)

def show_final(new_weights: list[int | float], x_train: list[int], y_train: list[int]):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 1.1)
    ax.set_xlabel('x2')
    ax.set_ylabel('x1')
    ax.set_zlabel('x3')
    for i in range(len(y_train)):
        if y_train[i] == 1:
            ax.scatter(x_train[i][2], x_train[i][1], x_train[i][3], c='green')
        else:
            ax.scatter(x_train[i][2], x_train[i][1], x_train[i][3], c='red')
    
    z = -1.6
    for x in range(260):
        ax.plot([-new_weights[-1][1]/new_weights[-1][2] * -2 - new_weights[-1][0] / new_weights[-1][2], 
            -new_weights[-1][1]/new_weights[-1][2] * 2 - new_weights[-1][0] / new_weights[-1][2]], [-2, 2], z, c='cyan')
        z += 0.01




def main():
    random.seed(5)
    w = [0.1, 0.1, 0.1, 0.1]

    x0 = 1
    x1 = [1.9, 0.5, -0.6, -0.8, -1.9]
    x2 = [-1.7, -0.8, 0.7, 0.8, 1.7]
    x3 = [-1.5, 0.9, 0.9, 0.7, -1.1]

    x_train = []
    y_train = []

    for i in range(5):
        for j in range(5):
            for k in range(5):
                x_train.append([x0, x1[i], x2[j], x3[k]])
                y_train.append(-1 if ((i == 1 and j == 4) or (i in (2, 3) and j in (2, 3, 4)) or (i == 4 and j != 0))  else 1)

    perc = Perceptron(w, x_train, y_train)
    perc.learning()
    show_final(perc.new_weights, x_train, y_train)


        
if __name__ == '__main__':
    main()
    plt.show()