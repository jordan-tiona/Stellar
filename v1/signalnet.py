import pybrain.structure as pbs
import pybrain.tools.customxml as pbxml

class SignalNet:

    # Length of historical data to use
    TIME_PERIOD = 30
    # Number of indicators per candle
    NUM_INDICATORS = 6
    # Ratio of hidden layer nodes to input layer nodes
    HIDDEN_RATIO = 1
    # Number of outputs
    NUM_OUTPUTS = 2

    def __init__(self, savedNetwork = None):
        if savedNetwork:
            self.network = pbxml.NetworkReader.readFrom(savedNetwork)
        else:
            self.network = pbs.networks.FeedForwardNetwork()
            inputLayer = pbs.modules.ReluLayer(self.TIME_PERIOD * self.NUM_INDICATORS)
            hiddenLayer = pbs.modules.SigmoidLayer(self.TIME_PERIOD * self.NUM_INDICATORS * self.HIDDEN_RATIO)
            outputLayer = pbs.modules.SigmoidLayer(self.NUM_OUTPUTS)

            self.network.addInputModule(inputLayer)
            self.network.addModule(hiddenLayer)
            self.network.addOutputModule(outputLayer)

            inputToHiddenWeights = pbs.connections.FullConnection(inputLayer, hiddenLayer)
            hiddenToOutputWeights = pbs.connections.FullConnection(hiddenLayer, outputLayer)

            self.network.addConnection(inputToHiddenWeights)
            self.network.addConnection(hiddenToOutputWeights)
            self.network.sortModules()

    def loadNetwork(self, filename):
        self.network = pbxml.NetworkReader.readFrom(filename)

    def saveNetwork(self, filename):
        pbxml.NetworkWriter.writeToFile(self.network, filename)

    def appendNetwork(self, filename):
        pbxml.NetworkWriter.appendToFile(self.network, filename)

    def clone(self):
        return self.network.copy()