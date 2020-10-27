import Utils as Utils
import numpy as numpy

data = Utils.getTrainingData()
trainingObjects = data[0]
Utils.normalize(trainingObjects)

y = [trainingObjects[i][data[1]] for i in range(data[2])]

for i in range(data[2]):
    del trainingObjects[i][-1]

singular = numpy.linalg.svd(trainingObjects)
F = singular[2] @ numpy.linalg.inv(singular[1]) @ singular[0].transpose()

