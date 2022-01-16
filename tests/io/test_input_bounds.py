import pytest
import tempfile

from omlt.io.input_bounds import write_input_bounds, load_input_bounds


def test_input_bounds_reader_writer_with_list():
    input_bounds = [(i*10.0, i*10.0 + 1.0) for i in range(10)]
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        write_input_bounds(f.name, input_bounds)

    bounds_back = load_input_bounds(f.name)

    for k, v in enumerate(input_bounds):
        assert bounds_back[k] == v



def test_input_bounds_reader_writer_with_dictionary():
    input_bounds = dict(
        ((i, i), (i*10.0, i*10.0 + 1.0))
        for i in range(10)
    )
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        write_input_bounds(f.name, input_bounds)

    bounds_back = load_input_bounds(f.name)

    for k, v in input_bounds.items():
        assert bounds_back[k] == v
