# 0.4

## Features
- Explicit handling of missing data. You can now provide e.g. a `Vector{Union{Missing, Float64}}` to a `to_sde(GP(...))` and expect sensible things to happen.
- Improved testing. There is a now a test suite for LGSSMs that can be found in `test/test_util.jl`.
