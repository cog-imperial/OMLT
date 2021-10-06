.PHONY: develop docs test

develop:
	python -m pip install -e .[testing]

docs:
	python -m tox -e docs

test:
	python -m tox