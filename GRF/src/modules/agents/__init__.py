REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_agent import ModularGatedCascadeCondNet
from .grnn_agent import GRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["soft"] = ModularGatedCascadeCondNet
REGISTRY["grnn"] = GRNNAgent
