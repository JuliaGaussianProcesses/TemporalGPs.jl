# 0.5.12

- A collection of examples of inference, and inference + learning, have been added.
    These are present in the examples directory, and appear in the tests.
    A separate CI run in which the examples are run has been added.
    This ensures that the examples run without error, but does not test correctness.
- Some test tolerances have been increased, as they were unnecessarily small.

# 0.4

## Features
- Explicit handling of missing data. You can now provide e.g. a `Vector{Union{Missing, Float64}}` to a `to_sde(GP(...))` and expect sensible things to happen.
- Improved testing. There is a now a test suite for LGSSMs that can be found in `test/test_util.jl`.
