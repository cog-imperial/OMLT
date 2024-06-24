import json
from pathlib import Path


def write_input_bounds(input_bounds_filename, input_bounds):
    """Write the specified input bounds to the given file."""
    input_bounds = _prepare_input_bounds(input_bounds)
    with Path(input_bounds_filename).open("w") as f:
        json.dump(input_bounds, f)


def load_input_bounds(input_bounds_filename):
    """Read the input bounds from the given file."""
    with Path(input_bounds_filename).open() as f:
        raw_input_bounds = json.load(f)

    return dict(_parse_raw_input_bounds(d) for d in raw_input_bounds)


def _prepare_input_bounds(input_bounds):
    if isinstance(input_bounds, list):
        return [
            {"key": i, "lower_bound": lb, "upper_bound": ub}
            for i, (lb, ub) in enumerate(input_bounds)
        ]
    # users should have passed a dict-like
    return [
        {"key": key, "lower_bound": lb, "upper_bound": ub}
        for key, (lb, ub) in input_bounds.items()
    ]


def _parse_raw_input_bounds(raw):
    key = raw["key"]
    lb = raw["lower_bound"]
    ub = raw["upper_bound"]
    if isinstance(key, list):
        key = tuple(key)
    return (key, (lb, ub))
