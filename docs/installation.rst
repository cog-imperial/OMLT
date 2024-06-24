Installation
============

OMLT requires Python >= 3.6. The most stable OMLT version can be installed using the PyPI package index. This will also install the required depencies. Simply run: ::

	pip install omlt

If using the latest un-released version, install from the github repository and install locally ::

	git clone https://github.com/cog-imperial/OMLT.git
	cd OMLT
	pip install .


Optional Requirements
---------------------

OMLT can import sequential Keras models which requires a working installation of tensorflow: ::

	pip install tensorflow

OMLT can also import neural network and gradient boosted tree models using ONNX. This requires installing the ONNX interface: ::

	pip install onnx

On Windows machines, the IPOPT solver executable is not installed with pyomo. See https://github.com/conda-forge/ipopt-feedstock/issues/55 for details. An older version of the solver can be installed with: ::

	conda install -c conda-forge ipopt=3.11.1
