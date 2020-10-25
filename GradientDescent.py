training = open('data/training.txt', 'r')
numberOfFeatures = int(training.readline())
numberOfObjects = int(training.readline())
objects = []

for line in training:
    objects.append(list(map(int, line.split())))

training.close()

