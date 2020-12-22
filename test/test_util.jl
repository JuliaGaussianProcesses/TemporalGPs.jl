using ChainRulesCore: backing
using TemporalGPs: Gaussian, correlate, decorrelate, harmonise



# Make FiniteDifferences work with some of the types in this package. Shame this isn't
# automated...

import FiniteDifferences: to_vec

function to_vec(x::Fill)
    x_vec, back_vec = to_vec(FillArrays.getindex_value(x))
    function Fill_from_vec(x_vec)
        return Fill(back_vec(x_vec), length(x))
    end
    return x_vec, Fill_from_vec
end

function to_vec(x::Union{Zeros, Ones})
    return Vector{eltype(x)}(undef, 0), _ -> x
end

function to_vec(x::Base.ReinterpretArray)
    return to_vec(collect(x))
end

function to_vec(::Missing)
    Missing_from_vec(::Any) = missing
    return Bool[], Missing_from_vec
end

function to_vec(x::T) where {T<:NamedTuple}
    isempty(fieldnames(T)) && throw(error("Expected some fields. None found."))
    vecs_and_backs = map(name->to_vec(getfield(x, name)), fieldnames(T))
    vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
    x_vec, back = to_vec(vecs)
    function namedtuple_to_vec(x′_vec)
        vecs′ = back(x′_vec)
        x′s = map((back, vec)->back(vec), backs, vecs′)
        return (; zip(fieldnames(T), x′s)...)
    end
    return x_vec, namedtuple_to_vec
end

function to_vec(x::T) where {T<:StaticArray}
    x_dense = collect(x)
    x_vec, back_vec = to_vec(x_dense)
    function StaticArray_to_vec(x_vec)
        return T(back_vec(x_vec))
    end
    return x_vec, StaticArray_to_vec
end

function to_vec(x::TemporalGPs.Gaussian)
    m_vec, m_from_vec = to_vec(x.m)
    P_vec, P_from_vec = to_vec(x.P)

    x_vec, x_back = to_vec((m_vec, P_vec))

    function Gaussian_from_vec(x_vec)
        mP_vec = x_back(x_vec)

        m = m_from_vec(mP_vec[1])
        P = P_from_vec(mP_vec[2])

        return TemporalGPs.Gaussian(m, P)
    end

    return x_vec, Gaussian_from_vec
end

function to_vec(gmm::TemporalGPs.GaussMarkovModel)
    A_vec, A_back = to_vec(gmm.A)
    a_vec, a_back = to_vec(gmm.a)
    Q_vec, Q_back = to_vec(gmm.Q)
    H_vec, H_back = to_vec(gmm.H)
    h_vec, h_back = to_vec(gmm.h)
    x0_vec, x0_back = to_vec(gmm.x0)

    gmm_vec, gmm_back = to_vec((A_vec, a_vec, Q_vec, H_vec, h_vec, x0_vec))

    function GaussMarkovModel_from_vec(gmm_vec)
        vecs = gmm_back(gmm_vec)
        A = A_back(vecs[1])
        a = a_back(vecs[2])
        Q = Q_back(vecs[3])
        H = H_back(vecs[4])
        h = h_back(vecs[5])
        x0 = x0_back(vecs[6])
        return TemporalGPs.GaussMarkovModel(A, a, Q, H, h, x0)
    end

    return gmm_vec, GaussMarkovModel_from_vec
end

function to_vec(model::TemporalGPs.LGSSM)
    gmm_vec, gmm_from_vec = to_vec(model.gmm)
    Σ_vec, Σ_from_vec = to_vec(model.Σ)

    model_vec, back = to_vec((gmm_vec, Σ_vec))

    function LGSSM_from_vec(model_vec)
        tmp = back(model_vec)
        gmm = gmm_from_vec(tmp[1])
        Σ = Σ_from_vec(tmp[2])
        return TemporalGPs.LGSSM(gmm, Σ)
    end

    return model_vec, LGSSM_from_vec
end

function to_vec(model::TemporalGPs.ScalarLGSSM)
    model_vec, lgssm_from_vec = to_vec(model.model)

    function ScalarLGSSM_from_vec(model_vec::AbstractVector{<:Real})
        return TemporalGPs.ScalarLGSSM(lgssm_from_vec(model_vec))
    end
    return model_vec, ScalarLGSSM_from_vec
end

function to_vec(model::TemporalGPs.CheckpointedLGSSM)
    model_vec, lgssm_from_vec = to_vec(model.model)

    function CheckpointedLGSSM_from_vec(model_vec::AbstractVector{<:Real})
        return TemporalGPs.CheckpointedLGSSM(lgssm_from_vec(model_vec))
    end
    return model_vec, CheckpointedLGSSM_from_vec
end

function to_vec(model::TemporalGPs.PosteriorLGSSM)
    model_vec, lgssm_from_vec = to_vec(model.model)
    function PosteriorLGSSM_from_vec(model_vec)
        return TemporalGPs.PosteriorLGSSM(lgssm_from_vec(model_vec))
    end
    return model_vec, PosteriorLGSSM_from_vec
end

function to_vec(X::BlockDiagonal)
    Xs = blocks(X)
    Xs_vec, Xs_from_vec = to_vec(Xs)

    function BlockDiagonal_from_vec(Xs_vec)
        Xs = Xs_from_vec(Xs_vec)
        return BlockDiagonal(Xs)
    end

    return Xs_vec, BlockDiagonal_from_vec
end

function to_vec(X::KroneckerProduct)
    A, B = getmatrices(X)
    A_vec, A_from_vec = to_vec(A)
    B_vec, B_from_vec = to_vec(B)
    X_vec, back = to_vec((A_vec, B_vec))

    function KroneckerProduct_from_vec(X_vec)
        (A_vec, B_vec) = back(X_vec)
        A = A_from_vec(A_vec)
        B = B_from_vec(B_vec)
        return A ⊗ B
    end

    return X_vec, KroneckerProduct_from_vec
end

to_vec(::Nothing) = Bool[], _ -> nothing

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
    @testset "to_vec(::GaussMarkovModel)" begin
        N = 11
        A = [randn(2, 2) for _ in 1:N]
        a = [randn(2) for _ in 1:N]
        Q = [randn(2, 2) for _ in 1:N]
        H = [randn(3, 2) for _ in 1:N]
        h = [randn(3) for _ in 1:N]
        x0 = TemporalGPs.Gaussian(randn(2), randn(2, 2))
        gmm = TemporalGPs.GaussMarkovModel(A, a, Q, H, h, x0)

        gmm_vec, gmm_from_vec = to_vec(gmm)
        @test gmm_vec isa Vector{<:Real}
        @test gmm_from_vec(gmm_vec) == gmm
    end
    @testset "to_vec(::LGSSM)" begin
        N = 11

        A = [randn(2, 2) for _ in 1:N]
        a = [randn(2) for _ in 1:N]
        Q = [randn(2, 2) for _ in 1:N]
        H = [randn(3, 2) for _ in 1:N]
        h = [randn(3) for _ in 1:N]
        x0 = TemporalGPs.Gaussian(randn(2), randn(2, 2))
        gmm = TemporalGPs.GaussMarkovModel(A, a, Q, H, h, x0)

        Σ = [randn(3, 3) for _ in 1:N]

        model = TemporalGPs.LGSSM(gmm, Σ)

        model_vec, model_from_vec = to_vec(model)
        @test model_from_vec(model_vec) == model

        @testset "ScalarLGSSM" begin
            model_vec, model_from_vec = to_vec(TemporalGPs.ScalarLGSSM(model))
            @test model_from_vec(model_vec) isa TemporalGPs.ScalarLGSSM
        end
    end
    @testset "to_vec(::BlockDiagonal)" begin
        Ns = [3, 5, 1]
        Xs = map(N -> randn(N, N), Ns)
        X = BlockDiagonal(Xs)

        X_vec, X_from_vec = to_vec(X)
        @test X_vec isa Vector{<:Real}
        @test X_from_vec(X_vec) == X
    end
    @testset "to_vec(::KroneckerProduct" begin
        A = randn(4, 5)
        B = randn(6, 7)
        X = A ⊗ B

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
function fd_isapprox(x::Real, y::Zero, rtol, atol)
    return fd_isapprox(x, zero(x), rtol, atol)
end

function fd_isapprox(x::T, y::T, rtol, atol) where {T}
    if !isstructtype(T)
        throw(ArgumentError("Non-struct types are not supported by this fallback."))
    end

    return all(n -> fd_isapprox(getfield(x, n), getfield(y, n), rtol, atol), fieldnames(T))
end

function adjoint_test(
    f, ȳ, x::Tuple, ẋ::Tuple;
    rtol=1e-9,
    atol=1e-9,
    fdm=central_fdm(5, 1),
    test=true,
    check_infers=true,
    context=Zygote.Context(),
)
    # Compute <Jᵀ ȳ, ẋ> = <x̄, ẋ> using Zygote.
    y, pb = Zygote._pullback(context, f, x...)
    x̄ = pb(ȳ)[2:end]
    inner_ad = dot(harmonise(Zygote.wrap_chainrules_input(x̄), ẋ)...)

    # Approximate <ȳ, J ẋ> = <ȳ, ẏ> using FiniteDifferences.
    ẏ = jvp(fdm, f, zip(x, ẋ)...)
    inner_fd = dot(harmonise(Zygote.wrap_chainrules_input(ȳ), ẏ)...)

    # @show inner_fd - inner_ad

    # Check that Zygote didn't modify the forwards-pass.
    test && @test fd_isapprox(y, f(x...), rtol, atol)

    # Check for approximate agreement in "inner-products".
    test && @test fd_isapprox(inner_ad, inner_fd, rtol, atol)

    # Check type inference if requested.
    if check_infers
        @code_warntype Zygote._pullback(context, f, x...)
        @code_warntype pb(ȳ)
        @inferred Zygote._pullback(context, f, x...)
        @inferred pb(ȳ)
    end

    return x̄
end

function adjoint_test(f, input::Tuple; kwargs...)
    Δoutput = rand_zygote_tangent(f(input...))
    return adjoint_test(f, Δoutput, input; kwargs...)
end

function adjoint_test(f, Δoutput, input::Tuple; kwargs...)
    ∂input = map(rand_tangent, input)
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

function check_adjoint_allocations(
    f, Δoutput, input::Tuple;
    context=Zygote.Context(),
    max_forward_allocs=0,
    max_backward_allocs=0,
)
    _, pb = _pullback(context, f, input...)
    @test allocs(
        @benchmark(_pullback($context, $f, $input...); samples=1, evals=1),
    ) <= max_forward_allocs
    @test allocs(@benchmark $pb($Δoutput) samples=1 evals=1) <= max_backward_allocs
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

"""
    ssm_interface_tests(
        rng::AbstractRNG, build_ssm, θ...;
        check_infers=true, check_rmad=false, rtol=1e-9, atol=1e-9,
    )

Basic consistency tests that any ssm should be able to satisfy. The purpose of these tests
is not to ensure correctness of any given implementation, only to ensure that it is self-
consistent and implements the required interface.

`build_ssm` should be a unary function that, when called on `θ`, should return an
`AbstractSSM`.
"""
function ssm_interface_tests(
    rng::AbstractRNG, ssm::AbstractSSM; check_infers=true, check_adjoints=true, kwargs...
)
    y_no_missing = rand(rng, ssm)

    @testset "basics" begin
        @inferred storage_type(ssm)
        @inferred dim_latent(ssm)
        (@inferred dim_obs(ssm)) == length(first(y_no_missing))
        @test length(ssm) == length(y_no_missing)
        @test eltype(ssm) == eltype(first(y_no_missing))
        @test is_of_storage_type(ssm, storage_type(ssm))
    end

    # Construct a missing-data example.
    y_missing = Vector{Union{Missing, eltype(y_no_missing)}}(undef, length(y_no_missing))
    y_missing .= y_no_missing
    y_missing[1] = missing
    if length(ssm) >= 3
        y_missing[3] = missing
    end
    missing_idx = length(ssm) >= 3 ? [1, 3] : [1]
    α_missing = whiten(ssm, y_missing)

    @testset "$(data.name)" for data in [
        (name="no-missings", y=y_no_missing),
        (name="with-missings", y=y_missing),
    ]
        _check_infers = data.name == "with-missings" ? false : check_infers

        y = data.y
        lml, α, xs = decorrelate(ssm, y)

        @testset "decorrelate" begin

            @test is_of_storage_type(α, storage_type(ssm))
            @test is_of_storage_type(xs, storage_type(ssm))

            @test lml == logpdf(ssm, y)
            @test α == whiten(ssm, y)
            @test xs == _filter(ssm, y)

            if _check_infers
                @inferred decorrelate(ssm, y)
                @inferred logpdf(ssm, y)
                @inferred whiten(ssm, y)
                @inferred _filter(ssm, y)
            end

            if check_adjoints
                adjoint_test(
                    decorrelate, (ssm, y);
                    check_infers=_check_infers, context=NoContext(), kwargs...,
                )
                Δoutput = rand_zygote_tangent((lml, α, xs))
                @testset "$(typeof(a1)), $(typeof(a2)), $(typeof(a3))" for
                    a1 in [nothing, Δoutput[1]],
                    a2 in [nothing, Δoutput[2]],
                    a3 in [nothing, Δoutput[3]]

                    adjoint_test(
                        decorrelate, (a1, a2, a3), (ssm, y);
                        check_infers=_check_infers, kwargs...,
                    )
                end
                adjoint_test(logpdf, (ssm, y); check_infers=_check_infers, kwargs...)
                adjoint_test(whiten, (ssm, y); check_infers=_check_infers, kwargs...)
                adjoint_test(_filter, (ssm, y); check_infers=_check_infers, kwargs...)
            end
        end

        data.name == "with-missings" && continue

        @testset "correlate" begin
            lml, y_cor, xs = correlate(ssm, α)

            @test is_of_storage_type(y_cor, storage_type(ssm))
            @test is_of_storage_type(xs, storage_type(ssm))

            @test lml ≈ logpdf(ssm, y)
            @test y_cor ≈ y
            @test y_cor == unwhiten(ssm, α)

            _lml, _y = logpdf_and_rand(rng, ssm)
            @test _lml ≈ logpdf(ssm, _y)
            @test length(_y) == length(y)

            if check_infers
                @inferred correlate(ssm, α)
                @inferred unwhiten(ssm, α)
                @inferred rand(rng, ssm)
                @inferred logpdf_and_rand(rng, ssm)
            end

            if check_adjoints
                Δoutput = rand_zygote_tangent((lml, y_cor, xs))
                @testset "$(typeof(a1)), $(typeof(a2)), $(typeof(a3))" for
                    a1 in [nothing, Δoutput[1]],
                    a2 in [nothing, Δoutput[2]],
                    a3 in [nothing, Δoutput[3]]

                    adjoint_test(
                        correlate, (a1, a2, a3), (ssm, α);
                        check_infers=check_infers, kwargs...,
                    )
                end
                adjoint_test(unwhiten, (ssm, α); check_infers=check_infers, kwargs...)
                adjoint_test(
                    ssm -> rand(deepcopy(rng), ssm), (ssm, );
                    check_infers=check_infers, kwargs...,
                )
                adjoint_test(
                    ssm -> logpdf_and_rand(deepcopy(rng), ssm), (ssm, );
                    check_infers=check_infers, kwargs...,
                )
            end
        end

        @testset "statistics" begin
            ds = marginals(ssm)
            @test vcat(mean.(ds)...) ≈ mean(ssm)
            if ds isa AbstractVector{<:Gaussian}
                @test vcat(diag.(cov.(ds))...) ≈ diag(cov(ssm))
            else
                @test vcat(var.(ds)) ≈ diag(cov(ssm))
            end

            @test isapprox(logpdf(Gaussian(mean(ssm), cov(ssm)), vcat(y...)), lml)

            check_infers && @inferred marginals(ssm)
            # adjoint_test(marginals, (ssm, ); check_infers=check_infers, kwargs...)
        end
    end
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::StaticArray)
    return map(x -> rand_tangent(rng, x), A)
end

FiniteDifferences.rand_tangent(::AbstractRNG, ::Base.OneTo) = Zero()

# Hacks to make rand_tangent play nicely with Zygote.
rand_zygote_tangent(A) = Zygote.wrap_chainrules_output(FiniteDifferences.rand_tangent(A))

Zygote.wrap_chainrules_output(x::Array) = map(Zygote.wrap_chainrules_output, x)

Zygote.wrap_chainrules_input(x::Array) = map(Zygote.wrap_chainrules_input, x)

function LinearAlgebra.dot(A::Composite, B::Composite)
    mutual_names = intersect(propertynames(A), propertynames(B))
    return sum(n -> dot(getproperty(A, n), getproperty(B, n)), mutual_names)
end
