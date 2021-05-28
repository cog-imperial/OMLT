import pyomo.environ as pyo
import pyomo.mpec as mpec
from pyomo.core.base.block import _BlockData, declare_custom_block
from .utils import _keras_sequential_to_dict

#Build the full-space representation of a neural net
def build_neural_net(block,n_inputs,n_outputs,n_nodes,w,b):
    #block sets
    block.INPUTS = pyo.Set(initialize=list(range(n_inputs)), ordered=True)
    block.NODES = pyo.Set(initialize=list(range(n_nodes)), ordered=True)
    block.INTERMEDIATE_NODES = pyo.Set(initialize=list(range(n_inputs, n_nodes)), ordered=True)
    block.OUTPUTS = pyo.Set(initialize=list(range(n_nodes, n_nodes+n_outputs)), ordered=True)

    # input variables
    block.x = pyo.Var(block.INPUTS)

    # pre-activation values
    block.zhat = pyo.Var(block.INTERMEDIATE_NODES)

    # post-activation values
    block.z = pyo.Var(block.NODES)

    # output variables
    block.y = pyo.Var(block.OUTPUTS)

    # set inputs
    @block.Constraint(block.INPUTS)
    def _inputs(m,i):
        return m.z[i] == m.x[i]

    #pre-activation logic
    @block.Constraint(block.INTERMEDIATE_NODES)
    def _linear(m,i):
        return m.zhat[i] == sum(w[i][j]*m.z[j] for j in w[i]) + b[i]

    #output logic
    @block.Constraint(block.OUTPUTS)
    def _outputs(m,i):
        return m.y[i] == sum(w[i][j]*m.z[j] for j in w[i]) + b[i]

class BaseNeuralNet:

    def __init__(self):
        self.n_nodes = 0
        self.n_inputs = 0
        self.n_outputs = 0
        self.w = dict()
        self.b = dict()

    def set_weights(self,w,b,n_inputs,n_outputs,n_nodes):
        assert n_nodes >= n_inputs
        assert len(w) == n_nodes + n_outputs - n_inputs
        assert len(b) == n_nodes + n_outputs - n_inputs
        self.w = w
        self.b = b
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = len(w) - self.n_outputs + self.n_inputs
        self.inputs = list(range(self.n_inputs))
        self.outputs = list(range(self.n_nodes, self.n_nodes+self.n_outputs))

class FullSpaceNet(BaseNeuralNet):
    """
        Builder class that creates a neural network surrogate for a Pyomo model.
        This class exposes the intermediate neural network variables as Pyomo variables and constraints.
    """
    def __init__(self,activation = pyo.tanh):
        super().__init__()
        self.activation = activation

    def _build_neural_net(self,block):
        build_neural_net(block,self.n_inputs,self.n_outputs,self.n_nodes,self.w,self.b)

    def _add_activation_constraint(self,block):
        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_activation(m,i):
            return m.z[i] == self.activation(m.zhat[i])

    def build(self,block):
        self._build_neural_net(block)
        self._add_activation_constraint(block)

class ReducedSpaceNet(BaseNeuralNet):
    """
        Builder class that creates a neural network surrogate for a Pyomo model.
        This class hides the intermediate nerual network variables inside Pyomo expressions.
    """
    def __init__(self,activation = pyo.tanh):
        super().__init__()
        self.activation = activation

    def _unpack_nn_expression(self,block,i):
        """
            Creates a Pyomo expression for output `i` of a neural network.  Uses recursion to build up the expression.
        """
        nodes_from = self.w[i]
        z_from = dict()
        for node in nodes_from:
            if node in self.w: #it's an output or intermediate node
                z_from[node] = self._unpack_nn_expression(block,node)
            else:              #it's an input node
                z_from[node] = block.x[node]

        if i in block.OUTPUTS: #don't apply activation to output
            z = sum(self.w[i][j]*z_from[j] for j in self.w[i]) + self.b[i]
        else:
            z = sum(self.w[i][j]*self.activation(z_from[j]) for j in self.w[i]) + self.b[i]

        return z

    def _build_neural_net(self,block):
        block.INPUTS = pyo.Set(initialize=self.inputs, ordered=True)
        block.OUTPUTS = pyo.Set(initialize=self.outputs, ordered=True)

        block.x = pyo.Var(block.INPUTS)
        block.y = pyo.Var(block.OUTPUTS)

        w = self.w
        b = self.b
        activation = self.activation

        def neural_net_constraint_rule(block,i):
            expr = self._unpack_nn_expression(block,i)
            return block.y[i] == expr

        block.NN = pyo.Constraint(block.OUTPUTS,rule=neural_net_constraint_rule)

    def build(self,block):
        self._build_neural_net(block)


class BigMReluNet(BaseNeuralNet):
    """
        Builder class for creating a MILP representation of a
        ReLU or LeakyReLU neural network on a Pyomo model.
    """
    def __init__(self,bigm = 1e6,leaky_alpha = None):
        super().__init__()
        self.M = bigm
        self.leaky_alpha = leaky_alpha
        if leaky_alpha is not None:
            raise NotImplementedError('LeakyReLU is not yet implemented')

    def _build_neural_net(self,block):
        build_neural_net(block,self.n_inputs,self.n_outputs,self.n_nodes,self.w,self.b)

    def _add_activation_constraint(self,block):
        # activation indicator q=0 means z=zhat (positive part of the hinge)
        # q=1 means we are on the zero part of the hinge
        block.q = pyo.Var(block.INTERMEDIATE_NODES, within=pyo.Binary)

        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_lower_bound(m,i):
            return m.z[i] >= 0

        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_zhat_bound(m,i):
            return m.z[i] >= m.zhat[i]

        # These are the activation binary constraints
        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_hat_positive(m,i):
            return m.z[i] <= m.zhat[i] + self.M*m.q[i]

        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_hat_negative(m,i):
            return m.z[i] <= self.M*(1.0-m.q[i])

    def _build_neural_net(self,block):
        build_neural_net(block,self.n_inputs,self.n_outputs,self.n_nodes,self.w,self.b)

    def build(self,block):
        self._build_neural_net(block)
        self._add_activation_constraint(block)

    #TODO
    def _perform_bounds_tightening(self,block):
        pass

    def _build_relaxation(self,block):
        pass

class ComplementarityReluNet(BaseNeuralNet):
    """
        Builder class for creating a MPEC representation of a
        ReLU or LeakyReLU neural network on a Pyomo model.
    """
    def __init__(self,leaky_alpha = None,transform = "mpec.simple_nonlinear"):
        super().__init__()

        if leaky_alpha is not None:
            raise NotImplementedError('LeakyReLU is not yet implemented')
        self.leaky_alpha = leaky_alpha
        self.transform = transform

    def _add_activation_constraint(self,block):
        @block.Complementarity(block.INTERMEDIATE_NODES)
        def _z_hat_positive(m,i):
            return mpec.complements((m.z[i] - m.zhat[i]) >= 0, m.z[i] >= 0)
        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(block)

    def _build_neural_net(self,block):
        build_neural_net(block,self.n_inputs,self.n_outputs,self.n_nodes,self.w,self.b)

    def build(self,block):
        self._build_neural_net(block)
        self._add_activation_constraint(block)

class TrainableNet(BaseNeuralNet):
    """
        Builds a Pyomo model that encodes the parameters of a neural net as variables.  The `TrainableNet` neural net builder
        can be used to train neural net parameters using Pyomo interfaced solvers.
    """
    def __init__(self,build_mode = FullSpaceNet(pyo.tanh)):
        super().__init__()
        self.build_mode = build_mode

    def build(self,block):

        block.PARAMETERS = pyo.Set(initialize = list(range(len(self.w))), ordered = True)
        block.w = pyo.Var(m.PARAMETERS,initialize = self.w)
        block.b = pyo.Var(m.PARAMETERS,initialize = self.b)

        self._build_neural_net(block)
        self.build_mode._add_activation_constraint(block) #could be full_space, complementarity, etc...


def ReLUNet(mode,*args,**kwargs):
    if mode == "bigm":
        return BigMReluMode(*args,**kwargs)
    elif mode == "complementarity":
        return ComplementarityReluMode(*args,**kwargs)
    else:
        error("Unknown ReLU mode provided.  Current supported modes are: `bigm` and `complementarity`")

#Entire neural net can be one dense constraint OR neural net can be encoded into layered logic
@declare_custom_block(name='NeuralNetBlock')
class NeuralBlockData(_BlockData):
    #NOTE: argument to constructor don't work for custom blocks
    # def __init__(self,mode = FullSpaceNet(activation = pyo.tanh)):
    #     self.mode = mode

    def __init__(self, component):#, mode = FullSpaceNet(activation = pyo.tanh)):
        super().__init__(component)
        self.neural_net = None

    def set_neural_net(self,net):
        self.neural_net = net

    def set_weights(self,w,b,n_inputs,n_outputs,n_nodes):
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
        self.neural_net.set_weights(w,b,n_inputs,n_outputs,n_nodes)


    def set_weights_keras(self,keras_model):
        """
            Unpack a keras model into dictionaries of weights and biases.  The dictionaries are used to build the underlying pyomo model.
        """
        w,b = _keras_sequential_to_dict(keras_model)  #TODO: sparse version
        n_inputs = len(keras_model.get_weights()[0])
        n_outputs = len(keras_model.get_weights()[-1])
        n_nodes = len(w) - self.n_outputs + self.n_inputs
        self.neural_net.set_weights(w,b,n_inputs,n_outputs,n_nodes)

    def build(self):
        """
            Build the pyomo block using the given neural network object.
        """
        self.neural_net.build(self)
