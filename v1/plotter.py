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
trainingData = data.downloadTrainingData(random.choice(symbols))
processedData = data.processTrainingData(trainingData)
sn = signalnet.SignalNet()
sn.loadNetwork('./saved/gen550.xml')

buy = []
sell = []

for day in range(signalnet.SignalNet.TIME_PERIOD - 1, len(processedData)):
    # Current day plus historical data
    currentRange = processedData[day - signalnet.SignalNet.TIME_PERIOD + 1 : day + 1]
    # Flatten list for SignalNet
    inputData = flatten(currentRange)

    outputs = (sn.network.activate(inputData))
    buy.append(outputs[0])
    sell.append(outputs[1])

opens = []
for day in trainingData[signalnet.SignalNet.TIME_PERIOD + 19:]:
    opens.append(day[0])

fig, axes = plt.subplots(nrows=2)
axes[0].plot(opens)
axes[1].plot(buy, color='g')
axes[1].plot(sell, color='r')
#plt.plot(opens)
#plt.plot(outputs)
plt.show()