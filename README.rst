.. image:: https://user-images.githubusercontent.com/282580/146039921-b3ea73af-7da3-47c1-bdfb-c40ad537a737.png
     :target: https://github.com/cog-imperial/OMLT
     :alt: OMLT
     :align: center
     :width: 200px

.. image:: https://github.com/cog-imperial/OMLT/workflows/CI/badge.svg?branch=main
     :target: https://github.com/cog-imperial/OMLT/actions?workflow=CI
     :alt: CI Status

.. image:: https://codecov.io/gh/cog-imperial/OMLT/branch/main/graph/badge.svg?token=9U7WLDINJJ
     :target: https://codecov.io/gh/cog-imperial/OMLT

.. image:: https://readthedocs.org/projects/omlt/badge/?version=latest
     :target: https://omlt.readthedocs.io/en/latest/?badge=latest
     :alt: Documentation Status
     
.. image:: https://user-images.githubusercontent.com/31448377/202018691-dfacb0f8-620d-4d48-b918-2fa8b8da3d26.png
     :target: https://www.coin-or.org/
     :alt: COIN
     :width: 130px


===============================================
OMLT: Optimization and Machine Learning Toolkit
===============================================

OMLT is a Python package for representing machine learning models (neural networks and gradient-boosted trees) within the Pyomo optimization environment. The package provides various optimization formulations for machine learning models (such as full-space, reduced-space, and MILP) as well as an interface to import sequential Keras and general ONNX models.

Please reference the `preprint <https://arxiv.org/abs/2202.02414>`_ of this software package as:

::

     @misc{ceccon2022omlt,
          title={OMLT: Optimization & Machine Learning Toolkit},
          author={Ceccon, F. and Jalving, J. and Haddad, J. and Thebelt, A. and Tsay, C. and Laird, C. D. and Misener, R.},
          year={2022},
          eprint={2202.02414},
          archivePrefix={arXiv},
          primaryClass={stat.ML}
     }

Documentation
==============
The latest OMLT documentation can be found at the `readthedocs page <https://omlt.readthedocs.io/en/latest/index.html#>`_. Additionally, much of the current functionality is demonstrated using Jupyter notebooks available in the  `notebooks folder <https://github.com/cog-imperial/OMLT/tree/main/docs/notebooks>`_.

Example
========

.. code-block:: Python

     import tensorflow
     import pyomo.environ as pyo
     from omlt import OmltBlock, OffsetScaling
     from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
     from omlt.io import load_keras_sequential

     #load a Keras model
     nn = tensorflow.keras.models.load_model('tests/models/keras_linear_131_sigmoid', compile=False)

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
     scaled_input_bounds = {0:(0,5)}

     #load the keras model into a network definition
     net = load_keras_sequential(nn,scaler,scaled_input_bounds)

     #multiple formulations of a neural network are possible
     #this uses the default NeuralNetworkFormulation object
     formulation = FullSpaceNNFormulation(net)

     #build the formulation on the OMLT block
     model.nn.build_formulation(formulation)

     #query inputs and outputs, as well as scaled inputs and outputs
     model.nn.inputs
     model.nn.outputs
     model.nn.scaled_inputs
     model.nn.scaled_outputs

     #connect pyomo model input and output to the neural network
     @model.Constraint()
     def connect_input(mdl):
         return mdl.input == mdl.nn.inputs[0]

     @model.Constraint()
     def connect_output(mdl):
         return mdl.output == mdl.nn.outputs[0]

     #solve an inverse problem to find that input that most closely matches the output value of 0.5
     model.obj = pyo.Objective(expr=(model.output - 0.5)**2)
     status = pyo.SolverFactory('ipopt').solve(model, tee=False)
     print(pyo.value(model.input))
     print(pyo.value(model.output))


Development
===========

OMLT uses :code:`tox`. to manage development tasks:

* :code:`tox -av` to list available tasks
* :code:`tox` to run tests
* :code:`tox -e lint` to check formatting and code styles
* :code:`tox -e format` to automatically format files
* :code:`tox -e docs` to build the documentation
* :code:`tox -e publish` to publish the package to PyPi

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
     - This work was funded by Sandia National Laboratories, Laboratory Directed Research and Development program

   * - |fracek|_
     - Francesco Ceccon
     - This work was funded by an Engineering & Physical Sciences Research Council Research Fellowship [GrantNumber EP/P016871/1]

   * - |carldlaird|_
     - Carl D. Laird
     - Initial work was funded by Sandia National Laboratories, Laboratory Directed Research and Development program. Current work supported by Carnegie Mellon University.

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

.. _carldlaird: https://github.com/carldlaird
.. |carldlaird| image:: https://avatars.githubusercontent.com/u/18519762?v=4
   :width: 80px

.. _tsaycal: https://github.com/tsaycal
.. |tsaycal| image:: https://avatars.githubusercontent.com/u/50914878?s=120&v=4
   :width: 80px

.. _thebtron: https://github.com/ThebTron
.. |thebtron| image:: https://avatars.githubusercontent.com/u/31448377?s=120&v=4
   :width: 80px
