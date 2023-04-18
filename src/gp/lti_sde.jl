"""
    LTISDE (Linear Time-Invariant Stochastic Differential Equation)

A lightweight wrapper around a `GP` `f` that tells this package to handle inference in `f`.
Can be constructed via the [`to_sde`](@ref) function.
"""
struct LTISDE{Tf<:GP, Tstorage<:StorageType} <: AbstractGP
    f::Tf
    storage::Tstorage
end

function to_sde(f::GP, storage_type=ArrayStorage(Float64))
    return LTISDE(f, storage_type)
end

storage_type(f::LTISDE) = f.storage

"""
    const FiniteLTISDE = FiniteGP{<:LTISDE}

A `FiniteLTISDE` is just a regular `FiniteGP` that happens to contain an `LTISDE`, as
opposed to any other `AbstractGP`, useful for dispatching.
"""
const FiniteLTISDE = FiniteGP{<:LTISDE}

# Deal with a bug in AbstractGPs.
function AbstractGPs.FiniteGP(f::LTISDE, x::AbstractVector{<:Real})
    return AbstractGPs.FiniteGP(f, x, convert(eltype(storage_type(f)), 1e-12))
end

# Implement the AbstractGP API.

function AbstractGPs.marginals(ft::FiniteLTISDE)
    return reduce(vcat, map(marginals, marginals(build_lgssm(ft))))
end

function AbstractGPs.mean_and_var(ft::FiniteLTISDE)
    ms = marginals(ft)
    return map(mean, ms), map(var, ms)
end

AbstractGPs.mean(ft::FiniteLTISDE) = first(mean_and_var(ft))

AbstractGPs.var(ft::FiniteLTISDE) = last(mean_and_var(ft))

AbstractGPs.cov(ft::FiniteLTISDE) = cov(FiniteGP(ft.f.f, ft.x, ft.Σy))

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteLTISDE)
    return destructure(ft.x, rand(rng, build_lgssm(ft)))
end

AbstractGPs.rand(ft::FiniteLTISDE) = rand(Random.GLOBAL_RNG, ft)

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteLTISDE, N::Int)
    return hcat([rand(rng, ft) for _ in 1:N]...)
end

AbstractGPs.rand(ft::FiniteLTISDE, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Real}) = _logpdf(ft, y)

function AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
    return _logpdf(ft, y)
end

function _logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
    return logpdf(build_lgssm(ft), observations_to_time_form(ft.x, y))
end

# Converting GPs into LGSSMs (Linear Gaussian State-Space Models).
function build_lgssm(f::LTISDE, x::AbstractVector, Σys::AbstractVector)
    m = get_mean(f)
    k = get_kernel(f)
    s = Zygote.literal_getfield(f, Val(:storage))
    As, as, Qs, emission_proj, x0 = lgssm_components(m, k, x, s)
    return LGSSM(
        GaussMarkovModel(Forward(), As, as, Qs, x0), build_emissions(emission_proj, Σys),
    )
end

function build_lgssm(ft::FiniteLTISDE)
    f = Zygote.literal_getfield(ft, Val(:f))
    x = Zygote.literal_getfield(ft, Val(:x))
    Σys = noise_var_to_time_form(x, Zygote.literal_getfield(ft, Val(:Σy)))
    return build_lgssm(f, x, Σys)
end

get_mean(f::LTISDE) = get_mean(Zygote.literal_getfield(f, Val(:f)))
get_mean(f::GP) = Zygote.literal_getfield(f, Val(:mean))

get_kernel(f::LTISDE) = get_kernel(Zygote.literal_getfield(f, Val(:f)))
get_kernel(f::GP) = Zygote.literal_getfield(f, Val(:kernel))

function build_emissions(
    (Hs, hs)::Tuple{AbstractVector, AbstractVector}, Σs::AbstractVector,
)
    Hst = _map(adjoint, Hs)
    return StructArray{get_type(Hst, hs, Σs)}((Hst, hs, Σs))
end

function get_type(Hs_prime, hs::AbstractVector{<:Real}, Σs)
    THs = eltype(Hs_prime)
    Ths = eltype(hs)
    TΣs = eltype(Σs)
    T = ScalarOutputLGC{THs, Ths, TΣs}
    return T
end

function get_type(Hs_prime, hs::AbstractVector{<:AbstractVector}, Σs)
    THs = eltype(Hs_prime)
    Ths = eltype(hs)
    TΣs = eltype(Σs)
    T = SmallOutputLGC{THs, Ths, TΣs}
    return T
end

@inline function Zygote.wrap_chainrules_output(x::NamedTuple)
    return map(Zygote.wrap_chainrules_output, x)
end

# Constructor for combining kernel and mean functions
function lgssm_components(
    ::ZeroMean, k::Kernel, t::AbstractVector, storage_type::StorageType
)
    return lgssm_components(k, t, storage_type)
end

function lgssm_components(
    m::AbstractGPs.MeanFunction, k::Kernel, t::AbstractVector, storage_type::StorageType
)
    m = collect(mean_vector(m, t)) # `collect` is needed as there are still issues with Zygote and FillArrays.
    As, as, Qs, (Hs, hs), x0 = lgssm_components(k, t, storage_type)
    hs = add_proj_mean(hs, m)

    return As, as, Qs, (Hs, hs), x0
end

# Either build a new vector or update an existing one with 
add_proj_mean(hs::AbstractVector{<:Real}, m) = hs .+ m
function add_proj_mean(hs::AbstractVector, m)
    return map((h, m) -> h + vcat(m, Zeros(length(h) - 1)), hs, m)
end

# Generic constructors for base kernels.

function broadcast_components((F, q, H)::Tuple, x0::Gaussian, t::AbstractVector{<:Real}, ::StorageType{T}) where {T}
    P = Symmetric(x0.P)
    t = vcat([first(t) - 1], t)
    As = _map(Δt -> time_exp(F, T(Δt)), diff(t))
    as = Fill(Zeros{T}(size(first(As), 1)), length(As))
    Qs = _map(A -> P - A * P * A', As)
    Hs = Fill(H, length(As))
    hs = Fill(zero(T), length(As))
    As, as, Qs, Hs, hs
end

function broadcast_components((F, q, H)::Tuple, x0::Gaussian, t::Union{StepRangeLen, RegularSpacing}, ::StorageType{T}) where {T}
    P = Symmetric(x0.P)
    A = time_exp(F, T(step(t)))
    As = Fill(A, length(t))
    as = @ignore_derivatives(Fill(Zeros{T}(size(F, 1)), length(t)))
    Q = Symmetric(P) - A * Symmetric(P) * A'
    Qs = Fill(Q, length(t))
    Hs = Fill(H, length(t))
    hs = Fill(zero(T), length(As))
    As, as, Qs, Hs, hs
end

function lgssm_components(
    k::SimpleKernel, t::AbstractVector{<:Real}, storage::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage)
    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    As, as, Qs, Hs, hs = broadcast_components(to_sde(k, storage), x0, t, storage)
    
    emission_projections = (Hs, hs)

    return As, as, Qs, emission_projections, x0
end

# Fallback definitions for most base kernels.
function to_sde(k::SimpleKernel, ::ArrayStorage{T}) where {T<:Real}
    F, q, H = to_sde(k, SArrayStorage(T))
    return collect(F), q, collect(H)
end

function stationary_distribution(k::SimpleKernel, ::ArrayStorage{T}) where {T<:Real}
    x = stationary_distribution(k, SArrayStorage(T))
    return Gaussian(collect(x.m), collect(x.P))
end

# Matern-1/2

function to_sde(::Matern12Kernel, ::SArrayStorage{T}) where {T<:Real}
    F = SMatrix{1, 1, T}(-1)
    q = convert(T, 2)
    H = SVector{1, T}(1)
    return F, q, H
end

function stationary_distribution(::Matern12Kernel, ::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{1, T}(0),
        SMatrix{1, 1, T}(1),
    )
end

# Matern - 3/2

function to_sde(::Matern32Kernel, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(3)
    F = SMatrix{2, 2, T}(0, -3, 1, -2λ)
    q = convert(T, 4 * λ^3)
    H = SVector{2, T}(1, 0)
    return F, q, H
end

function stationary_distribution(::Matern32Kernel, ::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{2, T}(0, 0),
        SMatrix{2, 2, T}(1, 0, 0, 3),
    )
end

# Matern - 5/2

function to_sde(::Matern52Kernel, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(5)
    F = SMatrix{3, 3, T}(0, 0, -λ^3, 1, 0, -3λ^2, 0, 1, -3λ)
    q = convert(T, 8 * λ^5 / 3)
    H = SVector{3, T}(1, 0, 0)
    return F, q, H
end

function stationary_distribution(::Matern52Kernel, ::SArrayStorage{T}) where {T<:Real}
    κ = 5 / 3
    m = SVector{3, T}(0, 0, 0)
    P = SMatrix{3, 3, T}(1, 0, -κ, 0, κ, 0, -κ, 0, 25)
    return Gaussian(m, P)
end

# Constant

function TemporalGPs.to_sde(::ConstantKernel, ::SArrayStorage{T}) where {T<:Real}
    F = SMatrix{1, 1, T}(0)
    q = convert(T, 0)
    H = SVector{1, T}(1)
    return F, q, H
end

function TemporalGPs.stationary_distribution(k::ConstantKernel, ::SArrayStorage{T}) where {T<:Real}
    return TemporalGPs.Gaussian(
        SVector{1, T}(0),
        SMatrix{1, 1, T}( T(only(k.c)) ),
    )
end

# Scaled

function to_sde(k::ScaledKernel, storage::StorageType{T}) where {T<:Real}
    _k = Zygote.literal_getfield(k, Val(:kernel))
    σ² = Zygote.literal_getfield(k, Val(:σ²))
    F, q, H = to_sde(_k, storage)
    σ = sqrt(convert(eltype(storage), only(σ²)))
    return F, σ^2 * q, σ * H
end

stationary_distribution(k::ScaledKernel, storage::StorageType) = stationary_distribution(Zygote.literal_getfield(k, Val(:kernel)), storage)

function lgssm_components(k::ScaledKernel, ts::AbstractVector, storage_type::StorageType)
    _k = Zygote.literal_getfield(k, Val(:kernel))
    σ² = Zygote.literal_getfield(k, Val(:σ²))
    As, as, Qs, emission_proj, x0 = lgssm_components(_k, ts, storage_type)
    σ = sqrt(convert(eltype(storage_type), only(σ²)))
    return As, as, Qs, _scale_emission_projections(emission_proj, σ), x0
end

function _scale_emission_projections((Hs, hs)::Tuple{AbstractVector, AbstractVector}, σ::Real)
    return _map(H->σ * H, Hs), _map(h->σ * h, hs)
end

function _scale_emission_projections((Cs, cs, Hs, hs), σ)
    return (Cs, cs, _map(H -> σ * H, Hs), _map(h -> σ * h, hs))
end

# Stretched

function to_sde(k::TransformedKernel{<:Kernel, <:ScaleTransform}, storage::StorageType)
    _k = Zygote.literal_getfield(k, Val(:kernel))
    s = Zygote.literal_getfield(Zygote.literal_getfield(k, Val(:transform)), Val(:s))
    F, q, H = to_sde(_k, storage)
    return F * only(s), q, H
end

stationary_distribution(k::TransformedKernel{<:Kernel, <:ScaleTransform}, storage::StorageType) = stationary_distribution(Zygote.literal_getfield(k, Val(:kernel)), storage)

function lgssm_components(
    k::TransformedKernel{<:Kernel, <:ScaleTransform},
    ts::AbstractVector,
    storage_type::StorageType,
)
    _k = Zygote.literal_getfield(k, Val(:kernel))
    s = Zygote.literal_getfield(Zygote.literal_getfield(k, Val(:transform)), Val(:s))
    return lgssm_components(_k, apply_stretch(s[1], ts), storage_type)
end

apply_stretch(a, ts::AbstractVector{<:Real}) = a * ts

apply_stretch(a, ts::StepRangeLen) = a * ts

function apply_stretch(a, ts::RegularSpacing)
    t0 = Zygote.literal_getfield(ts, Val(:t0))
    Δt = Zygote.literal_getfield(ts, Val(:Δt))
    N = Zygote.literal_getfield(ts, Val(:N))
    return RegularSpacing(a * t0, a * Δt, N)
end

# Product

function lgssm_components(k::KernelProduct, ts::AbstractVector, storage::StorageType)
    sde_kernels = to_sde.(k.kernels, Ref(storage))
    F_kernels = getindex.(sde_kernels, 1)
    F = foldl(_kron_add, F_kernels)
    q_kernels = getindex.(sde_kernels, 2)
    q = kron(q_kernels...)
    H_kernels = getindex.(sde_kernels, 3)
    H = kron(H_kernels...)

    x0_kernels = stationary_distribution.(k.kernels, Ref(storage))
    m_kernels = getproperty.(x0_kernels, :m)
    m = kron(m_kernels...)
    P_kernels = getproperty.(x0_kernels, :P)
    P = kron(P_kernels...)

    x0 = Gaussian(m, P)
    As, as, Qs, Hs, hs = broadcast_components((F, q, H), x0, ts, storage)
    emission_projections = (Hs, hs)
    return As, as, Qs, emission_projections, x0
end

_kron_add(A::AbstractMatrix, B::AbstractMatrix) = kron(A, I(size(B,1))) + kron(I(size(A,1)), B)
_kron_add(A::SMatrix{M,M}, B::SMatrix{N,N}) where {M, N} = kron(A, SMatrix{N, N}(I(N))) + kron(SMatrix{M,M}(I(M)), B)

# Sum

function lgssm_components(k::KernelSum, ts::AbstractVector, storage_type::StorageType)
    lgssms = lgssm_components.(k.kernels, Ref(ts), Ref(storage_type))
    As_kernels = getindex.(lgssms, 1)
    as_kernels = getindex.(lgssms, 2)
    Qs_kernels = getindex.(lgssms, 3)
    emission_proj_kernels = getindex.(lgssms, 4)
    x0_kernels = getindex.(lgssms, 5)
    
    As = _map(block_diagonal, As_kernels...)
    as = _map(vcat, as_kernels...)
    Qs = _map(block_diagonal, Qs_kernels...)
    emission_projections = _sum_emission_projections(emission_proj_kernels...)
    x0 = Gaussian(mapreduce(x -> getproperty(x, :m), vcat, x0_kernels), block_diagonal(getproperty.(x0_kernels, :P)...))
    return As, as, Qs, emission_projections, x0
end

function _sum_emission_projections(Hs_hs::Tuple{AbstractVector, AbstractVector}...)
    return map(vcat, first.(Hs_hs)...), sum(last.(Hs_hs))
end

function _sum_emission_projections(
    Cs_cs_Hs_hs::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}...,
)
    Cs = getindex.(Cs_cs_Hs_hs, 1)
    cs = getindex.(Cs_cs_Hs_hs, 2)
    Hs = getindex.(Cs_cs_Hs_hs, 3)
    hs = getindex.(Cs_cs_Hs_hs, 4)
    C = _map(vcat, Cs...)
    c = sum(cs)
    H = _map(block_diagonal, Hs...)
    h = _map(vcat, hs...)
    return C, c, H, h
end

Base.vcat(x::Zeros{T, 1}, y::Zeros{T, 1}) where {T} = Zeros{T}(length(x) + length(y))

function block_diagonal(As::AbstractMatrix{T}...) where {T}
    nblocks = length(As)
    sizes = size.(As)
    Xs = [i == j ? As[i] : Zeros{T}(sizes[j][1], sizes[i][2]) for i in 1:nblocks, j in 1:nblocks]
    return hvcat(ntuple(_ -> nblocks, nblocks), Xs...)
end

function ChainRulesCore.rrule(::typeof(block_diagonal), As::AbstractMatrix...)
    szs = size.(As)
    row_szs = (0, cumsum(first.(szs))...)
    col_szs = (0, cumsum(last.(szs))...)
    block_diagonal_rrule(Δ::AbstractThunk) = block_diagonal_rrule(unthunk(Δ))
    function block_diagonal_rrule(Δ)
        ΔAs = ntuple(length(As)) do i
            Δ[(row_szs[i]+1):row_szs[i+1], (col_szs[i]+1):col_szs[i+1]]
        end
        return NoTangent(), ΔAs...
    end
    return block_diagonal(As...), block_diagonal_rrule
end

function block_diagonal(As::SMatrix...)
    nblocks = length(As)
    sizes = size.(As)
    Xs = [i == j ? As[i] : zeros(SMatrix{sizes[j][1], sizes[i][2]}) for i in 1:nblocks, j in 1:nblocks]
    return hcat(Base.splat(vcat).(eachrow(Xs))...)
end

function ChainRulesCore.rrule(::typeof(block_diagonal), As::SMatrix...)
    szs = size.(As)
    row_szs = (0, cumsum(first.(szs))...)
    col_szs = (0, cumsum(last.(szs))...)
    function block_diagonal_rrule(Δ)
        ΔAs = ntuple(length(As)) do i
            Δ[SVector{szs[i][1]}((row_szs[i]+1):row_szs[i+1]), SVector{szs[i][2]}((col_szs[i]+1):col_szs[i+1])]
        end
        return NoTangent(), ΔAs...
    end
    return block_diagonal(As...), block_diagonal_rrule
end
