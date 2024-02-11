using AbstractGPs
using BlockDiagonals
using FillArrays
using LinearAlgebra
using Random: AbstractRNG, MersenneTwister
using StaticArrays
using StructArrays
using TemporalGPs
using TemporalGPs:
    AbstractLGSSM,
    ElementOfLGSSM,
    Gaussian,
    Forward,
    Reverse,
    GaussMarkovModel,
    LGSSM,
    ordering,
    SmallOutputLGC,
    posterior_and_lml,
    predict,
    conditional_rand,
    AbstractLGC,
    dim_out,
    dim_in,
    _filter,
    x0,
    scan_emit,
    ε_randn
using Test



function test_interface(
    rng::AbstractRNG, conditional::AbstractLGC, x::Gaussian;
    check_inferred=TEST_TYPE_INFER, check_adjoints=true, check_allocs=TEST_ALLOC, atol=1e-6, rtol=1e-6, kwargs...,
)
    x_val = rand(rng, x)
    y = conditional_rand(rng, conditional, x_val)

    @testset "rand" begin
        @test length(y) == dim_out(conditional)
        args = (TemporalGPs.ε_randn(rng, conditional), conditional, x_val)
        check_inferred && @inferred conditional_rand(args...)
    end

    @testset "predict" begin
        @test predict(x, conditional) isa Gaussian
        check_inferred && @inferred predict(x, conditional)
    end

    conditional isa ScalarOutputLGC || @testset "predict_marginals" begin
        @test predict_marginals(x, conditional) isa Gaussian
        pred = predict(x, conditional)
        pred_marg = predict_marginals(x, conditional)
        @test mean(pred_marg) ≈ mean(pred)
        @test diag(cov(pred_marg)) ≈ diag(cov(pred))
        @test cov(pred_marg) isa Diagonal
    end

    @testset "posterior_and_lml" begin
        args = (x, conditional, y)
        @test posterior_and_lml(args...) isa Tuple{Gaussian, Real}
        check_inferred && @inferred posterior_and_lml(args...)
    end
end

"""
    test_interface(
        rng::AbstractRNG, ssm::AbstractLGSSM;
        check_inferred=TEST_TYPE_INFER, check_adjoints=true, check_allocs=TEST_ALLOC, kwargs...
    )

Basic consistency tests that any LGSSM should be able to satisfy. The purpose of these tests
is not to ensure correctness of any given implementation, only to ensure that it is self-
consistent and implements the required interface.
"""
function test_interface(
    rng::AbstractRNG, ssm::AbstractLGSSM;
    check_inferred=TEST_TYPE_INFER, check_adjoints=true, check_allocs=TEST_ALLOC, rtol, atol, kwargs...
)
    y_no_missing = rand(rng, ssm)
    @testset "LGSSM interface" begin
        @testset "rand" begin
            @test is_of_storage_type(y_no_missing[1], storage_type(ssm))
            @test y_no_missing isa AbstractVector
            @test length(y_no_missing) == length(ssm)
            check_inferred && @inferred rand(rng, ssm)
            rng = MersenneTwister(123456)
        end

        @testset "basics" begin
            @inferred storage_type(ssm)
            @test length(ssm) == length(y_no_missing)
        end

        @testset "marginals" begin
            xs = marginals(ssm)
            @test is_of_storage_type(xs, storage_type(ssm))
            @test xs isa AbstractVector{<:Gaussian}
            @test length(xs) == length(ssm)
            check_inferred && @inferred marginals(ssm)
        end

        @testset "$(data.name)" for data in [
            (name="no-missings", y=y_no_missing),
            # (name="with-missings", y=y_missing),
        ]
            _check_inferred = data.name == "with-missings" ? false : check_inferred

            y = data.y
            @testset "logpdf" begin
                lml = logpdf(ssm, y)
                @test lml isa Real
                @test is_of_storage_type(lml, storage_type(ssm))
                _check_inferred && @inferred logpdf(ssm, y)
            end
            @testset "_filter" begin
                xs = _filter(ssm, y)
                @test is_of_storage_type(xs, storage_type(ssm))
                @test xs isa AbstractVector{<:Gaussian}
                @test length(xs) == length(ssm)
                _check_inferred && @inferred _filter(ssm, y)
            end
            @testset "posterior" begin
                posterior_ssm = posterior(ssm, y)
                @test length(posterior_ssm) == length(ssm)
                @test ordering(posterior_ssm) != ordering(ssm)
                _check_inferred && @inferred posterior(ssm, y)
            end
        end
    end
end
