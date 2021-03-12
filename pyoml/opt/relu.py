import pyomo.environ as pyo
from pyomo.mpec import *
from pyomo.core.base.block import _BlockData, declare_custom_block
from .utils import _sequential_to_dict

class BigMReluMode:

    def __init__(self,bigm = 1e6,leaky_alpha = None):
        if leaky_alpha is not None:
            raise NotImplementedError('LeakyReLU is not yet implemented')

        self.M = bigm
        self.leaky_alpha = leaky_alpha

    def build(self,block):
        """
            Builds a Pyomo model that captures the MIP representation of a
            ReLU or LeakyReLU neural network.
        """
        m = block
        self.block = block
        n_inputs = self.block.n_inputs
        n_outputs = self.block.n_outputs
        n_nodes = self.block.n_nodes

        m.INPUTS = pyo.Set(initialize=list(range(n_inputs)), ordered=True)
        m.NODES = pyo.Set(initialize=list(range(n_nodes)), ordered=True)
        m.INTERMEDIATE_NODES = pyo.Set(initialize=list(range(n_inputs, n_nodes)), ordered=True)
        m.OUTPUTS = pyo.Set(initialize=list(range(n_nodes, n_nodes+n_outputs)), ordered=True)

        # input variables / values
        # todo: remove bounds here and set them outside
        m.x = pyo.Var(m.INPUTS)

        # pre-activation values
        m.zhat = pyo.Var(m.INTERMEDIATE_NODES)

        # post-activation values
        m.z = pyo.Var(m.NODES)

        # activation indicator q=0 means z=zhat (positive part of the hinge)
        # q=1 means we are on the zero part of the hinge
        m.q = pyo.Var(m.INTERMEDIATE_NODES, within=pyo.Binary)

        # output variables
        m.y = pyo.Var(m.OUTPUTS)

        @m.Constraint(m.INPUTS)
        def _inputs(m,i):
            return m.z[i] == m.x[i]

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _linear(m,i):
            return m.zhat[i] == sum(m.w[i][j]*m.z[j] for j in m.w[i]) + m.b[i]

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_lower_bound(m,i):
            return m.z[i] >= 0

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_zhat_bound(m,i):
            return m.z[i] >= m.zhat[i]

        @m.Constraint(m.OUTPUTS)
        def _outputs(m,i):
            return m.y[i] == sum(m.w[i][j]*m.z[j] for j in m.w[i]) + m.b[i]

        # These are the activation binary constraints
        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_hat_positive(m,i):
            return m.z[i] <= m.zhat[i] + self.M*m.q[i]

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_hat_negative(m,i):
            return m.z[i] <= self.M*(1.0-m.q[i])

class ComplementarityReluMode:

    def __init__(self,leaky_alpha = None,transform = "mpec.simple_nonlinear"):
        if leaky_alpha is not None:
            raise NotImplementedError('LeakyReLU is not yet implemented')

        self.leaky_alpha = leaky_alpha
        self.transform = transform

    def build(self,block):
        """
            Builds a Pyomo model that captures the MPEC representation of a
            ReLU or LeakyReLU neural network.
        """
        m = block
        self.block = block
        n_inputs = self.block.n_inputs
        n_outputs = self.block.n_outputs
        n_nodes = self.block.n_nodes

        m.INPUTS = pyo.Set(initialize=list(range(n_inputs)), ordered=True)
        m.NODES = pyo.Set(initialize=list(range(n_nodes)), ordered=True)
        m.INTERMEDIATE_NODES = pyo.Set(initialize=list(range(n_inputs, n_nodes)), ordered=True)
        m.OUTPUTS = pyo.Set(initialize=list(range(n_nodes, n_nodes+n_outputs)), ordered=True)

        #input variables
        m.x = pyo.Var(m.INPUTS)

        # pre-activation values
        m.zhat = pyo.Var(m.INTERMEDIATE_NODES)

        # post-activation values
        m.z = pyo.Var(m.NODES)

        # output variables
        m.y = pyo.Var(m.OUTPUTS)

        @m.Constraint(m.INPUTS)
        def _inputs(m,i):
            return m.z[i] == m.x[i]

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _linear(m,i):
            return m.zhat[i] == sum(m.w[i][j]*m.z[j] for j in m.w[i]) + m.b[i]

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_lower_bound(m,i):
            return m.z[i] >= 0

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_zhat_bound(m,i):
            return m.z[i] >= m.zhat[i]

        @m.Constraint(m.OUTPUTS)
        def _outputs(m,i):
            return m.y[i] == sum(m.w[i][j]*m.z[j] for j in m.w[i]) + m.b[i]

        # These are the activation complementarity constraints
        @m.Complementarity(m.INTERMEDIATE_NODES)
        def _z_hat_positive(m,i):
            return complements((m.z[i] - m.zhat[i]) >= 0, m.z[i] >= 0)

        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(m)


@declare_custom_block(name='ReLUBlock')
class ReLUBlockData(_BlockData):

    def set_neural_data(self,n_inputs,n_outputs,n_nodes,w,b):
        """
        This functions sets up a Pyomo model to capture a ReLU neural network.

        Parameters
        ----------
           n_inputs : int
              The number of inputs to the network
           n_outputs : int
              The number of outputs from the network
           n_nodes : int
              The total number of nodes in the network (equal to n_inputs + # of intermediate nodes)
           w : dict of dict of floats
              The weights for the network. This is a sparse structure indexed by w[i][j] where
              w[i] is a dictionary of dictionaries, and w[i][j] is a dictionary of floats.
              All indices are integer, corresponding to the node numbers 0 -> n_nodes
           b : dict of floats
              The biases for the network. This is a dictionary of floats and should have length of
              n_nodes - n_inputs.
        """
        assert n_nodes >= n_inputs
        assert len(w) == n_nodes + n_outputs - n_inputs # first index of w
        assert len(b) == n_nodes + n_outputs - n_inputs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.w = w
        self.b = b

    def set_keras_sequential(self,keras_model):
        """
            Unpack a keras model into dictionaries of weights and biases.  The dictionaries are used to build the underlying pyomo model.
        """
        chain = keras_model
        w,b = _sequential_to_dict(chain)
        self.w = w
        self.b = b
        self.n_inputs = len(chain.get_weights()[0])
        self.n_outputs = len(chain.get_weights()[-1])
        self.n_nodes = len(w) - self.n_outputs + self.n_inputs


    def build(self,mode = BigMReluMode(bigm = 1e6)):
        """
            Build the pyomo block using the given mode (e.g. BigMReluMode or ComplementarityReluMode)
        """
        self.mode = mode
        self.mode.build(self)
