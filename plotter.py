import data
import random
import signalnet
import matplotlib.pyplot as plt
import numpy as np

def flatten(list):
    result = []
    for item in list:
        for value in item:
            result.append(value)

    return result

symbols = data.readTrainingSet('./training-set.txt')
trainingData = data.downloadTrainingData('GPK')
processedData = data.processTrainingData(trainingData)
sn = signalnet.SignalNet()
sn.loadNetwork('./saved/gen1650.xml')

buy = []
sell = []

for day in range(len(processedData)):
    inputData = processedData[day]

    outputs = (sn.network.activate(inputData))
    buy.append(outputs)

opens = []
for day in trainingData[data.BBANDS_TIME_PERIOD:]:
    opens.append(day[0])

fig, axes = plt.subplots(nrows=2)
axes[0].plot(opens)
axes[1].plot(buy, color='g')
plt.grid(True)
plt.show()