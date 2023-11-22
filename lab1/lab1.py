import random
import matplotlib.pyplot as plt

plt.xlim(-2, 2)
plt.xlabel('x1')
plt.ylim(-2, 2)
plt.ylabel('x2')
plt.title('Процесс обучения')

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])

random.seed(2) # Чтобы обеспечить повторяемость
LEARNING_RATE = 0.1
index_list = [x for x in range(25)] # Чтобы сделать порядок случайным

x_train = [] # Входы
y_train = [] # Выход (истина)
x0 = 1
x1 = [1.9, 0.5, -0.6, -0.8, -1.9]
x2 = [-1.7, -0.8, 0.7, 0.8, 1.7]

true_answers = 5
false_answers = 0

for i in range(5):
    y_train.extend([1]*true_answers)
    y_train.extend([-1]*false_answers)
    true_answers -= 1
    false_answers += 1
    for j in range(5):
        x_train.append((x0, x1[i], x2[j]))
y_train[12] = -1
 

w = [0.1, 0.1, 0.1] # Инициализируем «случайными» числами


# Для нейрона с n входами длины w and x должны быть равны n+1
def compute_output(w, x, y):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i] # Вычисление суммы взвешенных входов
    # print(w, x, z, -1 if z < 0 else 1, y)
    
    if z < 0: # Применение знаковой функции
        return -1
    else:
        return 1
old_weights = []
new_weights = []
# Цикл обучения персептрона
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list) # Сделать порядок случайным
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x, y) # Функция персептрона

        if y != p_out: # Обновить веса, когда неправильно
            old_weights.append(w.copy())
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            new_weights.append(w.copy())
            show_learning(w) # Показать обновлённые веса

for i in range(len(new_weights)):
    plt.plot([-2, 2], [-old_weights[i][1]/old_weights[i][2] * -2 - old_weights[i][0]/old_weights[i][2], 
                       -new_weights[i][1] / new_weights[i][2] * 2 - new_weights[i][0] / new_weights[i][2]], 
                       '--' if i == len(new_weights)-1 else '-')

plt.show()

plt.figure()
plt.xlim(-2, 2)
plt.xlabel('x1')
plt.ylim(-2, 2)
plt.ylabel('x2')
plt.title('Конечные значения весов после обучения')


plt.plot([-new_weights[-1][1]/new_weights[-1][2] * -2 - new_weights[-1][0] / new_weights[-1][2], 
          -new_weights[-1][1]/new_weights[-1][2] * 2 - new_weights[-1][0] / new_weights[-1][2]], [-2, 2])
for i in index_list:
    z = 0
    for j in range(len(new_weights[-1])):
        z += x_train[i][j] * new_weights[-1][j]
    if z < 0:
        plt.scatter(x_train[i][2], x_train[i][1],  marker='_', c='red')
    else:
        plt.scatter(x_train[i][2], x_train[i][1],  marker='+', c='green')
    

plt.show()