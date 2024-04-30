Jupyter Notebooks
===================

OMLT provides Jupyter notebooks to demonstrate its core capabilities. All notebooks can be found on the OMLT 
github `page <https://github.com/cog-imperial/OMLT/tree/main/docs/notebooks/>`_.

The first set of notebooks demonstrates the basic mechanics of OMLT and shows how to use it:

* `build_network.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/build_network.ipynb/>`_ shows how to manually create a `NetworkDefinition` object. This notebook is helpful for understanding the details of the internal layer structure that OMLT uses to represent neural networks. 

* `import_network.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/import_network.ipynb/>`_ shows how to import neural networks from Keras and PyTorch using ONNX interoperability. The notebook also shows how to import variable bounds from data.

* `neural_network_formulations.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/neural_network_formulations.ipynb>`_ showcases the different neural network formulations available in OMLT.

* `index_handling.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/index_handling.ipynb>`_ shows how to use `IndexMapper` to handle the mappings between indexes.

* `bo_with_trees.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/trees/bo_with_trees.ipynb>`_ incorporates gradient-boosted trees into a Bayesian optimization loop to optimize the Rosenbrock function.

* `linear_tree_formulations.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/trees/linear_tree_formulations.ipynb>`_ showcases the different linear model decision tree formulations available in OMLT.

The second set of notebooks gives application-specific examples:

* `mnist_example_dense.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/mnist_example_dense.ipynb>`_ trains a fully dense neural network on MNIST and uses OMLT to find adversarial examples.

* `mnist_example_convolutional.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/mnist_example_convolutional.ipynb>`_ trains a convolutional neural network on MNIST and uses OMLT to find adversarial examples.

* `graph_neural_network_formulation.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/graph_neural_network_formulation.ipynb>`_ transforms graph neural networks into OMLT and builds formulation to solve optimization problems.

* `auto-thermal-reformer.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/auto-thermal-reformer.ipynb>`_ develops a neural network surrogate (using sigmoid activations) with data from a process model built using `IDAES-PSE <https://github.com/IDAES/idaes-pse>`_.

* `auto-thermal-reformer-relu.ipynb <https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/auto-thermal-reformer-relu.ipynb>`_ develops a neural network surrogate (using ReLU activations) with data from a process model built using `IDAES-PSE <https://github.com/IDAES/idaes-pse>`_.
* 
