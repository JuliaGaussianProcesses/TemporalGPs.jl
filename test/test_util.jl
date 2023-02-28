using AbstractGPs
using BlockDiagonals
using ChainRulesCore: backing, ZeroTangent, NoTangent, Tangent
using ChainRulesTestUtils: ChainRulesTestUtils, test_approx, rand_tangent, test_rrule, @ignore_derivatives
using FiniteDifferences
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
    harmonise,
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
    _filter
using Test
using Zygote



# Make FiniteDifferences work with some of the types in this package. Shame this isn't
# automated...

import FiniteDifferences: to_vec

test_zygote_grad(f, args...; check_inferred=false, kwargs...) = test_rrule(Zygote.ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred, kwargs...) 

function test_zygote_grad_finite_differences_compatible(f, args...; kwargs...)
    x_vec, from_vec = to_vec(args)
    function finite_diff_compatible_f(x::AbstractVector)
        return @ignore_derivatives(f)(from_vec(x)...)
    end
    test_zygote_grad(finite_diff_compatible_f ⊢ NoTangent(), x_vec; kwargs...)
end

function to_vec(x::Fill)
    x_vec, back_vec = to_vec(FillArrays.getindex_value(x))
    function Fill_from_vec(x_vec)
        return Fill(back_vec(x_vec), axes(x))
    end
    return x_vec, Fill_from_vec
end

function to_vec(x::Union{Zeros, Ones})
    return Vector{eltype(x)}(undef, 0), _ -> x
end

# I'M OVERRIDING FINITEDIFFERENCES DEFINITION HERE. THIS IS BAD.
function to_vec(x::Diagonal)
    v, diag_from_vec = to_vec(x.diag)
    Diagonal_from_vec(v) = Diagonal(diag_from_vec(v))
    return v, Diagonal_from_vec
end

# function to_vec(x::T) where {T<:NamedTuple}
#     isempty(fieldnames(T)) && throw(error("Expected some fields. None found."))
#     vecs_and_backs = map(name->to_vec(getfield(x, name)), fieldnames(T))
#     vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
#     x_vec, back = to_vec(vecs)
#     function namedtuple_to_vec(x′_vec)
#         vecs′ = back(x′_vec)
#         x′s = map((back, vec)->back(vec), backs, vecs′)
#         return (; zip(fieldnames(T), x′s)...)
#     end
#     return x_vec, namedtuple_to_vec
# end

function to_vec(x::T) where {T<:StaticArray}
    x_dense = collect(x)
    x_vec, back_vec = to_vec(x_dense)
    function StaticArray_to_vec(x_vec)
        return T(back_vec(x_vec))
    end
    return x_vec, StaticArray_to_vec
end

function to_vec(x::Adjoint{<:Any, T}) where {T<:StaticVector}
    x_vec, back = to_vec(Matrix(x))
    Adjoint_from_vec(x_vec) = Adjoint(T(conj!(vec(back(x_vec)))))
    return x_vec, Adjoint_from_vec
end

function to_vec(::Tuple{})
    empty_tuple_from_vec(::AbstractVector) = ()
    return Bool[], empty_tuple_from_vec
end

function to_vec(x::StructArray{T}) where {T}
    x_vec, x_fields_from_vec = to_vec(StructArrays.components(x))
    function StructArray_from_vec(x_vec)
        x_field_vecs = x_fields_from_vec(x_vec)
        return StructArray{T}(Tuple(x_field_vecs))
    end
    return x_vec, StructArray_from_vec
end

function to_vec(x::TemporalGPs.LGSSM)
    x_vec, from_vec = to_vec((x.transitions, x.emissions))
    function LGSSM_from_vec(x_vec)
        (transition, emission) = from_vec(x_vec)
        return LGSSM(transition, emission)
    end
    return x_vec, LGSSM_from_vec
end

function to_vec(x::ElementOfLGSSM)
    x_vec, from_vec = to_vec((x.transition, x.emission))
    function ElementOfLGSSM_from_vec(x_vec)
        (transition, emission) = from_vec(x_vec)
        return ElementOfLGSSM(x.ordering, transition, emission)
    end
    return x_vec, ElementOfLGSSM_from_vec
end

to_vec(x::T) where {T} = generic_struct_to_vec(x)

# This is a copy from FiniteDifferences.jl without the try catch
function generic_struct_to_vec(x::T) where {T}
    Base.isstructtype(T) || throw(error("Expected a struct type"))
    isempty(fieldnames(T)) && return (Bool[], _ -> x) # Singleton types
    val_vecs_and_backs = map(name -> to_vec(getfield(x, name)), fieldnames(T))
    vals = first.(val_vecs_and_backs)
    backs = last.(val_vecs_and_backs)
    v, vals_from_vec = to_vec(vals)
    function structtype_from_vec(v::Vector{<:Real})
        val_vecs = vals_from_vec(v)
        vals = map((b, v) -> b(v), backs, val_vecs)
        return T(vals...)
    end
    return v, structtype_from_vec
end

to_vec(x::TemporalGPs.RectilinearGrid) = generic_struct_to_vec(x)

function to_vec(f::GP)
    gp_vec, t_from_vec = to_vec((f.mean, f.kernel))
    function GP_from_vec(v)
        m, k = t_from_vec(v)
        return GP(m, k)
    end
    return gp_vec, GP_from_vec
end

Base.zero(x::AbstractGPs.ZeroMean) = x
Base.zero(x::Kernel) = x
Base.zero(x::TemporalGPs.LTISDE) = x

function to_vec(X::BlockDiagonal)
    Xs = blocks(X)
    Xs_vec, Xs_from_vec = to_vec(Xs)

    function BlockDiagonal_from_vec(Xs_vec)
        Xs = Xs_from_vec(Xs_vec)
        return BlockDiagonal(Xs)
    end

    return Xs_vec, BlockDiagonal_from_vec
end

function to_vec(x::RegularSpacing)
    RegularSpacing_from_vec(v) = RegularSpacing(v[1], v[2], x.N)
    return [x.t0, x.Δt], RegularSpacing_from_vec
end

# Ensure that to_vec works for the types that we care about in this package.
@testset "custom FiniteDifferences stuff" begin
    @testset "NamedTuple" begin
        a, b = 5.0, randn(2)
        t = (a=a, b=b)
        nt_vec, back = to_vec(t)
        @test nt_vec isa Vector{Float64}
        @test back(nt_vec) == t
    end
    @testset "Fill" begin
        @testset "$(typeof(val))" for val in [5.0, randn(3)]
            x = Fill(val, 5)
            x_vec, back = to_vec(x)
            @test x_vec isa Vector{Float64}
            @test back(x_vec) == x
        end
    end
    @testset "Zeros{T}" for T in [Float32, Float64]
        x = Zeros{T}(4)
        x_vec, back = to_vec(x)
        @test x_vec isa Vector{eltype(x)}
        @test back(x_vec) == x
    end
    @testset "gaussian" begin
        @testset "Gaussian" begin
            x = TemporalGPs.Gaussian(randn(3), randn(3, 3))
            x_vec, back = to_vec(x)
            @test back(x_vec) == x
        end
    end
    @testset "to_vec(::SmallOutputLGC)" begin
        A = randn(2, 2)
        a = randn(2)
        Q = randn(2, 2)
        model = SmallOutputLGC(A, a, Q)
        model_vec, model_from_vec = to_vec(model)
        @test model_vec isa Vector{<:Real}
        @test model_from_vec(model_vec) == model
    end
    @testset "to_vec(::GaussMarkovModel)" begin
        N = 11
        A = [randn(2, 2) for _ in 1:N]
        a = [randn(2) for _ in 1:N]
        Q = [randn(2, 2) for _ in 1:N]
        H = [randn(3, 2) for _ in 1:N]
        h = [randn(3) for _ in 1:N]
        x0 = TemporalGPs.Gaussian(randn(2), randn(2, 2))
        gmm = TemporalGPs.GaussMarkovModel(Forward(), A, a, Q, x0)

        gmm_vec, gmm_from_vec = to_vec(gmm)
        @test gmm_vec isa Vector{<:Real}
        @test gmm_from_vec(gmm_vec) == gmm
    end
    @testset "StructArray" begin
        x = StructArray([Gaussian(randn(2), randn(2, 2)) for _ in 1:10])
        x_vec, x_from_vec = to_vec(x)
        @test x_vec isa Vector{<:Real}
        @test x_from_vec(x_vec) == x
    end
    @testset "to_vec(::LGSSM)" begin
        N = 11

        # Build GaussMarkovModel.
        A = [randn(2, 2) for _ in 1:N]
        a = [randn(2) for _ in 1:N]
        Q = [randn(2, 2) for _ in 1:N]
        x0 = Gaussian(randn(2), randn(2, 2))
        gmm = GaussMarkovModel(Forward(), A, a, Q, x0)

        # Build LGSSM.
        H = [randn(3, 2) for _ in 1:N]
        h = [randn(3) for _ in 1:N]
        Σ = [randn(3, 3) for _ in 1:N]
        model = TemporalGPs.LGSSM(gmm, StructArray(map(SmallOutputLGC, H, h, Σ)))

        model_vec, model_from_vec = to_vec(model)
        @test model_from_vec(model_vec) == model
    end
    @testset "to_vec(::BlockDiagonal)" begin
        Ns = [3, 5, 1]
        Xs = map(N -> randn(N, N), Ns)
        X = BlockDiagonal(Xs)

        X_vec, X_from_vec = to_vec(X)
        @test X_vec isa Vector{<:Real}
        @test X_from_vec(X_vec) == X
    end
end

my_zero(x) = zero(x)
my_zero(x::AbstractArray{<:Real}) = zero(x)
my_zero(x::AbstractArray) = map(my_zero, x)
my_zero(x::Tuple) = map(my_zero, x)

# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return fd_isapprox(x_fd, my_zero(x_fd), rtol, atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return all(fd_isapprox.(x_ad, x_fd, rtol, atol))
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([fd_isapprox(get(()->nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)])
end
function fd_isapprox(x::Gaussian, y::Gaussian, rtol, atol)
    return isapprox(x.m, y.m; rtol=rtol, atol=atol) &&
        isapprox(x.P, y.P; rtol=rtol, atol=atol)
end
function fd_isapprox(x::Real, y::ZeroTangent, rtol, atol)
    return fd_isapprox(x, zero(x), rtol, atol)
end
fd_isapprox(x::ZeroTangent, y::Real, rtol, atol) = fd_isapprox(y, x, rtol, atol)

function fd_isapprox(x_ad::T, x_fd::T, rtol, atol) where {T<:NamedTuple}
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end

function fd_isapprox(x::T, y::T, rtol, atol) where {T}
    if !isstructtype(T)
        throw(ArgumentError("Non-struct types are not supported by this fallback."))
    end

    return all(n -> fd_isapprox(getfield(x, n), getfield(y, n), rtol, atol), fieldnames(T))
end

function adjoint_test(
    f, ȳ, x::Tuple, ẋ::Tuple;
    rtol=1e-6,
    atol=1e-6,
    fdm=central_fdm(5, 1; max_range=1e-3),
    test=true,
    check_inferred=TEST_TYPE_INFER,
    context=Context(),
    kwargs...,
)
    # Compute <Jᵀ ȳ, ẋ> = <x̄, ẋ> using Zygote.
    y, pb = Zygote.pullback(f, x...)

    # Check type inference if requested.
    if check_inferred
        # @descend only works if you `using Cthulhu`.
        # @descend Zygote._pullback(context, f, x...)
        # @descend pb(ȳ)

        # @code_warntype Zygote._pullback(context, f, x...)
        # @code_warntype pb(ȳ)
        @inferred Zygote._pullback(context, f, x...)
        @inferred pb(ȳ)
    end
    x̄ = pb(ȳ)
    x̄_ad, ẋ_ad = harmonise(Zygote.wrap_chainrules_input(x̄), ẋ)
    inner_ad = dot(x̄_ad, ẋ_ad)
    
    # Approximate <ȳ, J ẋ> = <ȳ, ẏ> using FiniteDifferences.
    # x̄_fd = j′vp(fdm, f, ȳ, x...)
    ẏ = jvp(fdm, f, zip(x, ẋ)...)

    ȳ_fd, ẏ_fd = harmonise(Zygote.wrap_chainrules_input(ȳ), ẏ)
    inner_fd = dot(ȳ_fd, ẏ_fd)
    # Check that Zygote didn't modify the forwards-pass.
    test && @test fd_isapprox(y, f(x...), rtol, atol)

    # Check for approximate agreement in "inner-products".
    test && @test fd_isapprox(inner_ad, inner_fd, rtol, atol)

    return x̄
end

function adjoint_test(f, input::Tuple; kwargs...)
    Δoutput = rand_zygote_tangent(f(input...))
    return adjoint_test(f, Δoutput, input; kwargs...)
end

function adjoint_test(f, Δoutput, input::Tuple; kwargs...)
    ∂input = map(rand_zygote_tangent, input)
    return adjoint_test(f, Δoutput, input, ∂input; kwargs...)
end

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)

    # println("ad")
    # display(adjoint_ad)
    # println()

    # println("fd")
    # display(adjoint_fd)
    # println()

    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

using BenchmarkTools

# Also checks the forwards-pass because it's helpful.
function check_adjoint_allocations(
    f, Δoutput, input::Tuple;
    context=NoContext(),
    max_primal_allocs=0,
    max_forward_allocs=0,
    max_backward_allocs=0,
    kwargs...,
)
    _, pb = _pullback(context, f, input...)

    primal_allocs = allocs(@benchmark($f($input...); samples=1, evals=1))
    forward_allocs = allocs(
        @benchmark(_pullback($context, $f, $input...); samples=1, evals=1),
    )
    backward_allocs = allocs(@benchmark $pb($Δoutput) samples=1 evals=1)

    # primal_allocs = allocs(@benchmark($f($input...)))
    # forward_allocs = allocs(
    #     @benchmark(_pullback($context, $f, $input...)),
    # )
    # backward_allocs = allocs(@benchmark $pb($Δoutput))

    # @show primal_allocs
    # @show forward_allocs
    # @show backward_allocs

    @test primal_allocs <= max_primal_allocs
    @test forward_allocs <= max_forward_allocs
    @test backward_allocs <= max_backward_allocs
end

function check_adjoint_allocations(f, input::Tuple; kwargs...)
    return check_adjoint_allocations(f, rand_zygote_tangent(f(input...)), input; kwargs...)
end

function benchmark_adjoint(f, ȳ, args...; disp=false)
    disp && println("primal")
    primal = @benchmark($f($args...); samples=1, evals=1)
    if disp
        display(primal)
        println()
    end

    disp && println("pullback generation")
    forward_pass = @benchmark(Zygote.pullback($f, $args...); samples=1, evals=1)
    if disp
        display(forward_pass)
        println()
    end

    y, back = Zygote.pullback(f, args...)

    disp && println("pullback evaluation")
    reverse_pass = @benchmark($back($ȳ); samples=1, evals=1)
    if disp
        display(reverse_pass)
        println()
    end

    return primal, forward_pass, reverse_pass
end

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
        if check_adjoints
            test_zygote_grad(
                conditional_rand, args...;
                check_inferred, rtol, atol,
            )
        end
        if check_allocs
            check_adjoint_allocations(conditional_rand, args; kwargs...)
        end
    end

    @testset "predict" begin
        @test predict(x, conditional) isa Gaussian
        check_inferred && @inferred predict(x, conditional)
        check_adjoints && adjoint_test(predict, (x, conditional); kwargs...)
        check_allocs && check_adjoint_allocations(predict, (x, conditional); kwargs...)
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
        if check_adjoints
            (Δx, Δlml) = rand_zygote_tangent(posterior_and_lml(args...))
            ∂args = map(rand_tangent, args)
            adjoint_test(posterior_and_lml, (Δx, Δlml), args, ∂args)
            adjoint_test(posterior_and_lml, (Δx, nothing), args, ∂args)
            adjoint_test(posterior_and_lml, (nothing, Δlml), args, ∂args)
            adjoint_test(posterior_and_lml, (nothing, nothing), args, ∂args)
        end
        if check_allocs
            (Δx, Δlml) = rand_zygote_tangent(posterior_and_lml(args...))
            check_adjoint_allocations(posterior_and_lml, (Δx, Δlml), args; kwargs...)
            check_adjoint_allocations(posterior_and_lml, (nothing, Δlml), args; kwargs...)
            check_adjoint_allocations(posterior_and_lml, (Δx, nothing), args; kwargs...)
            check_adjoint_allocations(posterior_and_lml, (nothing, nothing), args; kwargs...)
        end
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

    @testset "rand" begin
        @test is_of_storage_type(y_no_missing[1], storage_type(ssm))
        @test y_no_missing isa AbstractVector
        @test length(y_no_missing) == length(ssm)
        check_inferred && @inferred rand(rng, ssm)
        if check_adjoints
            # adjoint_test(
                # ssm -> rand(MersenneTwister(123456), ssm), (ssm,);
                # check_inferred, kwargs...
            # ) # TODO fix this test
            # test_zygote_grad(
                # ssm -> rand(MersenneTwister(123456), ssm), ssm;
                # check_inferred, rtol, atol,
            # )
        end
        if check_allocs
            check_adjoint_allocations(rand, (rng, ssm); kwargs...)
        end
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
        if check_adjoints
            test_zygote_grad(marginals, ssm; check_inferred, rtol, atol)
        end
        if check_allocs
            check_adjoint_allocations(marginals, (ssm, ); kwargs...)
        end
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

        # Hack to only run the AD tests if requested.
        @testset "adjoints" for _ in (check_adjoints ? [1] : [])
            adjoint_test(logpdf, (ssm, y); check_inferred=_check_inferred, kwargs...)
            adjoint_test(_filter, (ssm, y); check_inferred=_check_inferred, kwargs...)
            adjoint_test(posterior, (ssm, y); check_inferred=_check_inferred, kwargs...)

            if check_allocs
                check_adjoint_allocations(logpdf, (ssm, y); kwargs...)
                check_adjoint_allocations(_filter, (ssm, y); kwargs...)
                check_adjoint_allocations(posterior, (ssm, y); kwargs...)
            end
        end
    end
end

# This is unfortunately needed to make ChainRulesTestUtils comparison works.
# See https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/271
Base.zero(::Forward) = Forward()
Base.zero(::Reverse) = Reverse()

_diag(x) = diag(x)
_diag(x::Real) = x

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::StaticArray)
    return map(x -> rand_tangent(rng, x), A)
end

FiniteDifferences.rand_tangent(::AbstractRNG, ::Base.OneTo) = ZeroTangent()

# Hacks to make rand_tangent play nicely with Zygote.
rand_zygote_tangent(A) = Zygote.wrap_chainrules_output(FiniteDifferences.rand_tangent(A))

Zygote.wrap_chainrules_output(x::Array) = map(Zygote.wrap_chainrules_output, x)

function Zygote.wrap_chainrules_input(x::Array)
    return map(Zygote.wrap_chainrules_input, x)
end

function LinearAlgebra.dot(A::Tangent, B::Tangent)
    mutual_names = intersect(propertynames(A), propertynames(B))
    if length(mutual_names) == 0
        return 0
    else
        return sum(n -> dot(getproperty(A, n), getproperty(B, n)), mutual_names)
    end
end

function ChainRulesTestUtils.test_approx(actual::Tangent{T}, expected::StructArray, msg=""; kwargs...) where {T<:StructArray}
    return test_approx(actual.components, expected; kwargs...)
end