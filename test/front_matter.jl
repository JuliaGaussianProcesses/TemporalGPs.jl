
using AbstractGPs,
    BlockDiagonals,
    FillArrays,
    LinearAlgebra,
    KernelFunctions,
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
    RegularInTime

ENV["TESTING"] = "TRUE"

# GROUP is an env variable from CI which can take the following values
# ["test util", "test models" "test models-lgssm" "test gp" "test space_time"]
# Select any of this to test a particular aspect.
# To test everything, simply set GROUP to "all"
# ENV["GROUP"] = "test gp"
const GROUP = get(ENV, "GROUP", "all")

const TEST_TYPE_INFER = false # Test type stability over the tests
const TEST_ALLOC = false # Test allocations over the tests

include("test_util.jl")
include(joinpath("models", "model_test_utils.jl"))
