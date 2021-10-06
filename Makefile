.PHONY: develop test

develop:
	python -m pip install -e .[testing]

test:
	python -m tox