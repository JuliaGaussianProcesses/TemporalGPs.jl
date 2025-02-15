# 0.7.1

time_exp has been removed in favour of assuming that whichever AD library is being used can
successfully AD through the matrix exponential. Guard rails to prevent mis-use of previous
rule have been remved.

# 0.7

Mooncake.jl (and probably Enzyme.jl) is now able to differentiate everything in
TemporalGPs.jl _reasonably_ efficiently, and only requires a single rule (for time_exp).
This is in stark contrast with Zygote.jl, which required roughly 2.5k lines to achieve
reasonable performance. This code was not robust, required maintenance from time-to-time,
and generally made making progress on improvements to this library hard to make.
Consequently, in this version of TemporalGPs, we have removed all Zygote-related
functionality, and now recommend that Mooncake.jl (or perhaps Enzyme.jl) is used to
differentiate code in this package. In some places Mooncake.jl achieves worse performance
than Zygote.jl, but it is worth it for the amount of code that has been removed.

If you wish to use Zygote + TemporalGPs, you should restrict yourself to the 0.6 series of
this package.

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
