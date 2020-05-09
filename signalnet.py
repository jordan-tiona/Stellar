from pybrain.structure.modules import LSTMLayer
from pybrain.structure.connections import FullConnection
from pybrain.structure.modules.softsign import SoftSignLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkReader, NetworkWriter
from numpy.matlib import sqrt

class SignalNet:

    NUM_INDICATORS = 6
    NUM_LSTM_CELLS = 30
    NUM_OUTPUTS = 1

    def __init__(self):
        self.network = buildNetwork(self.NUM_INDICATORS, self.NUM_LSTM_CELLS,  self.NUM_OUTPUTS, hiddenclass=LSTMLayer, outclass=SoftSignLayer, peepholes=True, bias=False, outputbias=False, recurrent=True)
        self.scaleParams()

    def scaleParams(self):
        for con in self.network._containerIterator():
            factor = 1.0 / sqrt(con.indim)
            for param in range(len(con.params)):
                con.params[param] *= factor

    def saveNetwork(self, filename):
        NetworkWriter.writeToFile(self.network, filename)

    def loadNetwork(self, filename):
        self.network = NetworkReader.readFrom(filename)