import random as random
import Utils as Utils

trainingObjects = Utils.getTrainingData()

EPSILON = 0.000001

weights = [1 / (2 * random.randint(1, i)) for i in range(1, 10)]
step = 1


def gradientDescent():
    return 0
