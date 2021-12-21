.. image:: https://user-images.githubusercontent.com/282580/146039921-b3ea73af-7da3-47c1-bdfb-c40ad537a737.png
     :target: https://github.com/cog-imperial/OMLT
     :alt: OMLT
     :align: center
     :width: 200px

.. image:: https://github.com/cog-imperial/OMLT/workflows/CI/badge.svg?branch=main
     :target: https://github.com/cog-imperial/OMLT/actions?workflow=CI
     :alt: CI Status


===============================================
OMLT: Optimization and Machine Learning Toolkit
===============================================

OMLT is a Python package for representing machine learning models (such as neural networks) within the Pyomo optimization environment. The package provides various formulations for representing machine-learning models (such as full-space, reduced-space, and MILP), as well as an interface to import sequential Keras models.


Examples
========

.. code-block:: Python

   import tensorflow 
   import pyomo.environ as pyo
   from omlt import OmltBlock, OffsetScaling
   from omlt.neuralnet import FullSpaceContinuousFormulation, ReducedSpaceContinuousFormulation
   from omlt.neuralnet import ReLUBigMFormulation
   from omlt.neuralnet import load_keras_sequential

   #load a Keras model
   nn = tensorflow.keras.models.load_model('omlt/tests/models/keras_linear_131_sigmoid', compile=False)

   #create a Pyomo model with an OMLT block
   model = pyo.ConcreteModel()
   model.nn = OmltBlock()

   #the neural net contains one input and one output
   model.input = pyo.Var()
   model.output = pyo.Var()

   #apply simple offset scaling for the input and output
   scale_x = (1, 0.5)       #(mean,stdev) of the input
   scale_y = (-0.25, 0.125) #(mean,stdev) of the output
   scaler = OffsetScaling(offset_inputs=[scale_x[0]],
                       factor_inputs=[scale_x[1]],
                       offset_outputs=[scale_y[0]],
                       factor_outputs=[scale_y[1]])

   #provide bounds on the input variable (e.g. from training)
   input_bounds = [(0,5),]

   #load the keras model into a network definition
   net = load_keras_sequential(nn,scaler,input_bounds)

   #multiple neural network formulations are possible
   #hides the intermediate variables from the optimizer
   formulation = ReducedSpaceContinuousFormulation(net)

   #encodes intermediate neural network variables
   #formulation = FullSpaceContinuousFormulation(net)

   #encodes intermediate relu activations using binary variables
   #this requires a neural network with only linear and relu activations
   #formulation = ReLUBigMFormulation(net)

   #build the formulation on the OMLT block
   model.nn.build_formulation(formulation, input_vars=[model.input], output_vars=[model.output])

   #query inputs and outputs, as well as scaled inputs and outputs 
   model.nn.inputs_list
   model.nn.outputs_list 
   model.nn.scaled_inputs_list 
   model.nn.scaled_outputs_list

   #solve an inverse problem to find that input that most closely matches the output value of 0.5
   model.obj = pyo.Objective(expr=(model.output - 0.5)**2)
   status = pyo.SolverFactory('ipopt').solve(model, tee=False)
   print(pyo.value(model.input))
   print(pyo.value(model.output))


Development
===========

OMLT uses `tox` to manage development tasks:

* `tox -av` to list available tasks
* `tox` to run tests
* `tox -e lint` to check formatting and code styles
* `tox -e format` to automatically format files
* `tox -e docs` to build the documentation
* `tox -e publish` to publish the package to PyPi

Contributors
============

.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - GitHub
     - Name
     - Acknowledgements

   * - |jalving|_
     - Jordan Jalving 
     - Sandia National Laboratories is a multimission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC, a wholly owned  subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525

   * - |fracek|_
     - Francesco Ceccon
     - This work was funded by an Engineering & Physical Sciences Research Council Research Fellowship [GrantNumber EP/P016871/1]
     
   * - |tsaycal|_
     - Calvin Tsay
     - This work was funded by an Engineering & Physical Sciences Research Council Research Fellowship [GrantNumber EP/T001577/1], with additional support from an Imperial College Research Fellowship.
     
   * - |thebtron|_
     - Alexander Thebelt
     - This work was supported by BASF SE, Ludwigshafen am Rhein.


.. _jalving: https://github.com/jalving
.. |jalving| image:: https://avatars1.githubusercontent.com/u/16785413?s=120&v=4
   :width: 80px

.. _fracek: https://github.com/fracek
.. |fracek| image:: https://avatars1.githubusercontent.com/u/282580?s=120&v=4
   :width: 80px
   
.. _tsaycal: https://github.com/tsaycal
.. |tsaycal| image:: https://avatars.githubusercontent.com/u/50914878?s=120&v=4
   :width: 80px
   
.. _thebtron: https://github.com/ThebTron
.. |thebtron| image:: https://avatars.githubusercontent.com/u/31448377?s=120&v=4
   :width: 80px
