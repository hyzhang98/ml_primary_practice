import numpy as np
import random


class Machine:
    def __init__(self):
        self.arms = []
        self.arms.append((-1, 2))
        self.arms.append((0, 2))
        self.arms.append((-2, 3))
        self.arms.append((-0.1, 0.6))
        self.arms.append((2, 4))

    def choose(self, k):
        return np.random.normal(self.arms[k][0], self.arms[k][1], 1)[0]


def begin():
    machine = Machine()
    expectations = np.zeros(5)
    count = np.zeros(5) + 1
    n = 0
    # while True:
    for j in range(0, 1000):
        num = random.randint(0, 9)
        if num == 0:
            index = random.choice(range(0, 5))
            value = machine.choose(index)
            c = count[index]
            expectations[index] = expectations[index] * (c / (c + 1)) + value / (c + 1)
            count[index] += 1
        else:
            index = np.argmax(expectations)
            choices = [index]
            for i in range(index, len(expectations)):
                if index != i and expectations[index] == expectations[i]:
                    choices.append(i)
            choice = random.choice(choices)
            value = machine.choose(choice)
            c = count[choice]
            expectations[choice] = expectations[choice] * (c / (c + 1)) + value / (c + 1)
            count[choice] += 1
        if j % 50 == 0:
            print(expectations)


begin()
