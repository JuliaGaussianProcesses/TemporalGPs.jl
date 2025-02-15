
using Aqua,
    AbstractGPs,
    BlockDiagonals,
    FillArrays,
    JET,
    JuliaFormatter,
    KernelFunctions,
    LinearAlgebra,
    Mooncake,
    Random,
    StaticArrays,
    StructArrays,
    TemporalGPs,
    Test

using AbstractGPs: var
using Mooncake.TestUtils: test_rule
using TemporalGPs:
    AbstractLGSSM,
    _filter,
    Gaussian,
    x0,
    fill_in_missings,
    replace_observation_noise_cov,
    scan_emit,
    transform_model_and_obs,
    RectilinearGrid,
    RegularInTime,
    posterior_and_lml,
    predict,
    predict_marginals,
    step_marginals,
    step_logpdf,
    step_filter,
    step_rand,
    invert_dynamics,
    step_posterior,
    storage_type,
    is_of_storage_type,
    ArrayStorage,
    SArrayStorage,
    SmallOutputLGC,
    LargeOutputLGC,
    ScalarOutputLGC,
    Forward,
    Reverse,
    ordering

ENV["TESTING"] = "TRUE"

# GROUP is an env variable from CI which can take the following values
# ["test util", "test models" "test models-lgssm" "test gp" "test space_time"]
# Select any of this to test a particular aspect.
# To test everything, simply set GROUP to "all"
# ENV["GROUP"] = "test gp"
const GROUP = get(ENV, "GROUP", "all")

# Some test-local type piracy. ConstantKernel doesn't have a default constructor, so
# Mooncake's testing functionality doesn't work with it properly. To resolve this, I just
# add a default-style constructor here.
@eval function KernelFunctions.ConstantKernel{P}(c::Vector{P}) where {P<:Real}
    return $(Expr(:new, :(ConstantKernel{P}), :c))
end

@eval function PeriodicKernel{P}(c::Vector{P}) where {P<:Real}
    return $(Expr(:new, :(PeriodicKernel{P}), :c))
end

include("test_util.jl")
include(joinpath("models", "model_test_utils.jl"))
