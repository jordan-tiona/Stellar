from gatrainer import GATrainer
from signalnet import SignalNet
from random import Random
import data

print('Reading training set...')
symbols = data.readTrainingSet('./training-set.txt')

rand = Random()
rand.seed()
trainer = GATrainer()

while True:
    symbol = rand.choice(symbols)
    print('Downloading training data for ' + symbol)
    try:
        trainingData = data.downloadTrainingData(symbol)
    except:
        print('Error downloading training data, skipping')
        continue

    print('Processing training data...')
    processedData = data.processTrainingData(trainingData)
    trainingData = trainingData[data.BBANDS_TIME_PERIOD:]
    print('Running simulation...')
    trainer.runSimulation(processedData, trainingData)
    print('--------------------------------------------------')
    trainer.logStats()
    if trainer.genNumber % 50 == 0:
        print('Saving best performer...')
        trainer.saveBest()
    print('--------------------------------------------------\n')
    trainer.breedNextGeneration()