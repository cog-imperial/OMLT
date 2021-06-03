.PHONY: develop test

develop:
	pip install -e .[testing]

test:
	tox