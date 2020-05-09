from signalnet import SignalNet
import random
import statistics
import os
from numpy.matlib import randn

class GATrainer:

    POP_SIZE = 150
    INITIAL_MR = 0.05
    MR_DECAY_RATE = 50
    MINIMUM_MR = 0.0005
    BREEDING_POOL_SIZE = 15
    population = []
    genNumber = 0

    means = []
    highs = []

    def __init__(self):
        for _i in range(self.POP_SIZE):
            net = SignalNet()
            self.population.append({'net': net, 'outputs': [], 'score': 0.0, 'holding': False, 'side': '', 'enterPrice': 0.0})

    @staticmethod
    def flatten(list):
        result = []
        for item in list:
            for value in item:
                result.append(value)

        return result

    def runSimulation(self, processedData, trainingData):
        """Simulate trading for each SignalNet, updating their scores"""
        assert len(processedData) == len(trainingData)
        for day in range(len(processedData)):
            for sn in self.population:
                output = sn['net'].network.activate(processedData[day])
                sn['outputs'].append(output)

                # Positive output signals a buy
                if output > 0.0:
                    if (sn['holding'] == False):
                        # Worst case scenario, we buy at open plus 2%
                        sn['enterPrice'] = trainingData[day][0] * 1.02
                        sn['side'] = 'long'
                        sn['holding'] = True
                    else:
                        if (sn['side'] == 'short'):
                            # Cover short sell at low plus 2%
                            exitPrice = trainingData[day][2] * 1.02
                            sn['score'] += (sn['enterPrice'] - exitPrice) / sn['enterPrice']
                            sn['holding'] = False
                    
                # Negative output signals a sell
                if output <= 0.0:
                    if sn['holding'] == False:
                        # Sell short for open less 2%
                        sn['enterPrice'] = trainingData[day][0] * 0.98
                        sn['side'] = 'short'
                        sn['holding'] = True
                    else:
                        if (sn['side'] == 'long'):
                            # Sell for high less 2%
                            exitPrice = trainingData[day][1] * 0.98
                            sn['score'] += (exitPrice - sn['enterPrice']) / sn['enterPrice'] 
                            sn['holding'] = False
                """
                # If we're at the end of the training data and still holding, sell it
                if (day == len(processedData) - 1):
                    if sn['holding'] == True:
                        if (sn['side'] == 'long'):
                            exitPrice = trainingData[day][1] * 0.98
                            sn['score'] += (exitPrice - sn['enterPrice']) / sn['enterPrice']
                            sn['holding'] = False
                        else:
                            exitPrice = trainingData[day][2] * 1.02
                            sn['score'] += (sn['enterPrice'] - exitPrice) / sn['enterPrice']
                            sn['holding'] = False
                """
        self.genNumber += 1

    # pylint: disable=unsubscriptable-object
    def breedNextGeneration(self):
        """Breed top performers to create the next generation of SignalNets"""
        # Mutation rate is inversely proportional to generation number, starting at 3%
        mutationRate = self.MINIMUM_MR + ((self.INITIAL_MR - self.MINIMUM_MR) * self.MR_DECAY_RATE / (self.genNumber + self.MR_DECAY_RATE)) if self.genNumber != 0 else self.INITIAL_MR

        print('Breeding next generation (MR: {:.2f}%)'.format((mutationRate * 100.0)))

        # Sort population by score
        self.population.sort(reverse = True, key = lambda sn: sn['score'])
        # Pick top performers
        pool = self.population[0:self.BREEDING_POOL_SIZE]
        children = []
        rand = random.Random()
        rand.seed()
        # Breed POP_SIZE new children from pool
        for _c in range(self.POP_SIZE):
            sample = rand.sample(pool, 2)
            first = sample[0]
            second = sample[1]
            child = self.__breed(first['net'], second['net'], mutationRate)
            children.append({'net': child, 'outputs': [], 'score': 0.0, 'holding': False, 'boughtPrice': 0.0})

        self.population = children


    @staticmethod
    def __breed(first, second, mutationRate):
        """Breeds two networks and returns their child"""
        rand = random.Random()
        rand.seed()
        child = SignalNet()
        for p in range(len(first.network.params)):
            # If mutating this weight then leave it at the already randomized number
            if (rand.random() > mutationRate):
                # Randomly pick weight from first or second parent
                if (rand.random() >= 0.5):
                    child.network.params[p] = first.network.params[p]
                else:
                    child.network.params[p] = second.network.params[p]
        
        return child

    def saveBest(self):
        self.population.sort(reverse = True, key = lambda sn: sn['score'])
        filename = './saved/gen{}.xml'.format(self.genNumber)
        self.population[0]['net'].saveNetwork(filename)

    def logStats(self):
        # Sort population by score
        self.population.sort(reverse = True, key = lambda sn: sn['score'])

        scores = []
        for sn in self.population:
            scores.append(sn['score'])

        mean = statistics.mean(scores)
        self.means.append(mean)
        self.means = self.means[-50:]
        meansma = sum(self.means) / len(self.means)
        high = max(scores)
        self.highs.append(high)
        self.highs = self.highs[-50:]
        highsma = sum(self.highs) / len(self.highs)
        sdn = statistics.stdev(self.means) if len(self.means) > 1 else 0.0

        print('\t\tGeneration #' + str(self.genNumber))
        print('\tHighest Score:'.ljust(23) + '{:>.2%}'.format(high).ljust(8))
        print('\tAverage Score:'.ljust(23) + '{:>.2%}'.format(mean).ljust(8))
        print('\tStandard Deviation:'.ljust(23) + '{:>.2}'.format(sdn).ljust(8))
        print('\tMeans SMA:'.ljust(23) + '{:>.2%}'.format(meansma).ljust(8))
        print('\tHighs SMA:'.ljust(23) + '{:>.2%}'.format(highsma).ljust(8))
