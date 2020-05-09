import requests
import time
import signalnet
import gatrainer
import random

BBANDS_TIME_PERIOD = 20
RSI_TIME_PERIOD = 14

class Indicator:
    O = 0
    C = 1
    LB = 2
    MB = 3
    UB = 4
    RSI = 5
    OBV = 6

def readTrainingSet(filename):
    """Read list of symbols to train on from newline-seperated file"""
    with open(filename) as file:
        data = file.read()
        return data.split('\n')

def apiQuery(url, params):
    response = requests.get(url, params)

    # Rate limit reached, pause for a bit then throw excpetion
    if response.status_code == 429:
        print('-- Finnhub rate limit reached, pausing --')
        time.sleep(15)
        print('-- Resuming --')
        raise Exception('Finnhub: rate limit reached')

    f = response.json()
    # Symbol either not supported by Finnhub or some other error occurred
    if f['s'] == 'no_data':
        print('-- Finnhub responded with no data')
        raise Exception('Finnhub: no_data')

    return f

def downloadTrainingData(symbol):
    """Download training data from Finnhub"""

    now = int(time.time())
    lastYear = now - (365 * 24 * 60 * 60)
    url = 'https://finnhub.io/api/v1/indicator'

    # Daily and Bollinger Bands
    params = {
        'symbol': symbol,
        'resolution': 'D',
        'from': lastYear,
        'to': now,
        'indicator': 'bbands',
        'timeperiod': BBANDS_TIME_PERIOD,
        'token': 'bqat2uvrh5r8t7qnc9k0'
    }

    f = apiQuery(url, params)

    # RSI
    params['indicator'] = 'rsi'
    params['timeperiod'] = RSI_TIME_PERIOD

    f['rsi'] = apiQuery(url, params)['rsi']

    # OBV
    params['indicator'] = 'obv'

    f['obv'] = apiQuery(url, params)['obv']

    # Zip values together: {[o, c, lb, mb, ub, rsi, obv]}
    trainingData = []
    for (o, c, lb, mb, ub, rsi, obv) in zip(f['o'], f['c'], f['lowerband'], f['middleband'], f['upperband'], f['rsi'], f['obv']):
        trainingData.append([o, c, lb, mb, ub, rsi, obv])

    """
    # Zip values together: {[o, h, l, c, v, sma, ema, rsi], [...]}
    trainingData = []
    for (o, h, l, c, v, sma, ema, rsi) in zip(f['o'], f['h'], f['l'], f['c'], f['v'], f['sma'], f['ema'], f['rsi']):
        trainingData.append([o, h, l, c, v, sma, ema, rsi])
    """

    return trainingData


def processTrainingData(inData):
    """Process raw training data into relative values for signal network inputs"""
    previous = inData[0]
    processedData = []
    for day in inData:
        # Close and OBV are relative to previous value
        c   = (day[Indicator.C] - previous[Indicator.C]) / previous[Indicator.C] if previous[Indicator.C] != 0 else 0.0
        obv = (day[Indicator.OBV] - previous[Indicator.OBV]) / previous[Indicator.OBV] if previous[Indicator.OBV] != 0 else 0.0
        # BBands are relative to current close value
        lb = (day[Indicator.LB] - day[Indicator.C]) / day[Indicator.C] if day[Indicator.C] != 0 else 0.0
        mb = (day[Indicator.MB] - day[Indicator.C]) / day[Indicator.C] if day[Indicator.C] != 0 else 0.0
        ub = (day[Indicator.UB] - day[Indicator.C]) / day[Indicator.C] if day[Indicator.C] != 0 else 0.0
        # RSI is normalized to between 0 and 1
        rsi = day[Indicator.RSI] / 100.0

        processedData.append([c, lb, mb, ub, rsi, obv])
        previous = day

    # Strip first BBANDS_TIME_PERIOD days, they were only there for BBANDS calculation
    processedData = processedData[BBANDS_TIME_PERIOD:]
    return processedData


print('Reading training set...')
symbols = readTrainingSet('./training-set.txt')

rand = random.Random()
rand.seed()
trainer = gatrainer.GATrainer()

while True:
    symbol = rand.choice(symbols)
    print('Downloading training data for ' + symbol)
    try:
        trainingData = downloadTrainingData(symbol)
    except:
        print('Error downloading training data, skipping')
        continue

    print('Processing training data...')
    processedData = processTrainingData(trainingData)
    trainingData = trainingData[BBANDS_TIME_PERIOD:]
    print('Running simulation...')
    trainer.runSimulation(processedData, trainingData)
    print('--------------------------------------------------')
    trainer.logStats()
    if trainer.genNumber % 50 == 0:
        print('Saving best performer...')
        trainer.saveBest()
    print('--------------------------------------------------\n')
    trainer.breedNextGeneration()
