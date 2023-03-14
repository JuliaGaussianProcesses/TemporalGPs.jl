"""
    LTISDE (Linear Time-Invariant Stochastic Differential Equation)

A lightweight wrapper around a `GP` `f` that tells this package to handle inference in `f`.
Can be constructed via the `to_sde` function.
"""
struct LTISDE{Tf<:GP{<:AbstractGPs.ZeroMean}, Tstorage<:StorageType} <: AbstractGP
    f::Tf
    storage::Tstorage
end

function to_sde(f::GP{<:AbstractGPs.ZeroMean}, storage_type=ArrayStorage(Float64))
    return LTISDE(f, storage_type)
end

storage_type(f::LTISDE) = f.storage

"""
    const FiniteLTISDE = FiniteGP{<:LTISDE}

A `FiniteLTISDE` is just a regular `FiniteGP` that happens to contain an `LTISDE`, as
opposed to any other `AbstractGP`.
"""
const FiniteLTISDE = FiniteGP{<:LTISDE}

# Deal with a bug in AbstractGPs.
function FiniteGP(f::LTISDE, x::AbstractVector{<:Real})
    return FiniteGP(f, x, convert(eltype(storage_type(f)), 1e-12))
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
    k = get_kernel(f)
    s = Zygote.literal_getfield(f, Val(:storage))
    As, as, Qs, emission_proj, x0 = lgssm_components(k, x, s)
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



# Generic constructors for base kernels.

function lgssm_components(
    k::SimpleKernel, t::AbstractVector{<:Real}, storage::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage)
    P = x0.P
    F, _, H = to_sde(k, storage)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    t = vcat([first(t) - 1], t)
    As = _map(Δt -> time_exp(F, T(Δt)), diff(t))
    as = Fill(Zeros{T}(size(first(As), 1)), length(As))
    Qs = _map(A -> Symmetric(P) - A * Symmetric(P) * A', As)
    Hs = Fill(H, length(As))
    hs = Fill(zero(T), length(As))
    emission_projections = (Hs, hs)

    return As, as, Qs, emission_projections, x0
end

function lgssm_components(
    k::SimpleKernel, t::Union{StepRangeLen, RegularSpacing}, storage_type::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage_type)
    P = x0.P
    F, _, H = to_sde(k, storage_type)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    A = time_exp(F, T(step(t)))
    As = Fill(A, length(t))
    as = @ignore_derivatives(Fill(Zeros{T}(size(F, 1)), length(t)))
    Q = Symmetric(P) - A * Symmetric(P) * A'
    Qs = Fill(Q, length(t))
    Hs = Fill(H, length(t))
    hs = Fill(zero(T), length(As))
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

# Cosine

function to_sde(::CosineKernel, ::SArrayStorage{T}) where {T}
    F = SMatrix{2, 2, T}(0, 1, -1, 0)
    q = zero(T)
    H = SVector{2, T}(1, 0)
    return F, q, H
end
    
function stationary_distribution(::CosineKernel, ::SArrayStorage{T}) where {T<:Real}
    m = SVector{2, T}(0, 0)
    P = SMatrix{2, 2, T}(1, 0, 0, 1)
    return Gaussian(m, P)
end

# Approximate Periodic Kernel
# The periodic kernel is approximated by a sum of cosine kernels with different frequencies.
function lgssm_components(kernel::PeriodicKernel, ts::AbstractVector, storage::StorageType{T}) where {T}
    r = kernel.r
    length(r) == 1 || error("the state-space version of the `PeriodicKernel` only supports 1-dimensional inputs.")
    l⁻² = inv(4 * only(r))
    x0 = stationary_distribution(CosineKernel(), storage)
    P = x0.P
    F, _, H = to_sde(CosineKernel(), storage)
    # We follow "State Space approximation of Gaussian Processes for time series forecasting"
    # by Alessio Benavoli1 and Giorgio Corani and take 7 Cosine Kernel terms
    N = 7
    qs = ntuple(N) do i
        (1 + (i !== 1) ) * besseli(i - 1, l⁻²) / exp(l⁻²)
    end
    Fs = _map(q -> q * F, qs)
    Ps = _map(q -> q * P, qs) 
    t = vcat([first(ts) - 1], t)
    nt = length(diff(t))
    As = _map(F -> _map(Δt -> time_exp(F, T(Δt)), diff(t)), Fs)
    as = Fill(Fill(Zeros{T}(size(first(first(As)), 1)), nt), N)
    Qs = _map((P, A) -> _map(A -> Symmetric(P) - A * Symmetric(P) * A', A), Ps, As)
    H = Fill(H, nt)
    h = Fill(zero(T), nt)
    As = reduce(As) do As, A
        _map(blk_diag, As, A)
    end
    as = reduce(as) do as, a
        _map(vcat, as, a)
    end
    Qs = reduce(Qs) do Qs, Q
        _map(blk_diag, Qs, Q)
    end
    Hs = reduce(Fill(H, N)) do Hs, H
        _map(vcat, Hs, H)
    end
    m = reduce(Fill(x0.m, N)) do ms, m
        _map(vcat, ms, m)
    end
    P = reduce(Ps) do Ps, P
        _map(blk_diag, Ps, P)
    end
    x0 = Gaussian(m, P)
    return As, as, Qs, (Hs, h), x0
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

# Sum

function lgssm_components(k::KernelSum, ts::AbstractVector, storage_type::StorageType)
    return reduce(map(kernel -> lgssm_components(kernel, ts, storage_type), k.kernels)) do (As_l, as_l, Qs_l, emission_proj_l, x0_l), (As_r, as_r, Qs_r, emission_proj_r, x0_r)
        As = _map(blk_diag, As_l, As_r)
        as = _map(vcat, as_l, as_r)
        Qs = _map(blk_diag, Qs_l, Qs_r)
        emission_projections = _sum_emission_projections(emission_proj_l, emission_proj_r)
        x0 = Gaussian(vcat(x0_l.m, x0_r.m), blk_diag(x0_l.P, x0_r.P))

        return As, as, Qs, emission_projections, x0
    end
end

function _sum_emission_projections(
    (Hs_l, hs_l)::Tuple{AbstractVector, AbstractVector},
    (Hs_r, hs_r)::Tuple{AbstractVector, AbstractVector},
)
    return map(vcat, Hs_l, Hs_r), hs_l + hs_r
end

function _sum_emission_projections(
    (Cs_l, cs_l, Hs_l, hs_l)::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector},
    (Cs_r, cs_r, Hs_r, hs_r)::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector},
)
    Cs = _map(vcat, Cs_l, Cs_r)
    cs = cs_l + cs_r
    Hs = _map(blk_diag, Hs_l, Hs_r)
    hs = _map(vcat, hs_l, hs_r)
    return Cs, cs, Hs, hs
end

Base.vcat(x::Zeros{T, 1}, y::Zeros{T, 1}) where {T} = Zeros{T}(length(x) + length(y))

function blk_diag(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    return hvcat(
        (2, 2),
        A, zeros(T, size(A, 1), size(B, 2)), zeros(T, size(B, 1), size(A, 2)), B,
    )
end

function ChainRulesCore.rrule(::typeof(blk_diag), A, B)
    blk_diag_rrule(Δ::AbstractThunk) = blk_diag_rrule(unthunk(Δ))
    function blk_diag_rrule(Δ)
        ΔA = Δ[1:size(A, 1), 1:size(A, 2)]
        ΔB = Δ[size(A, 1)+1:end, size(A, 2)+1:end]
        return NoTangent(), ΔA, ΔB
    end
    return blk_diag(A, B), blk_diag_rrule
end

function blk_diag(A::SMatrix{DA, DA, T}, B::SMatrix{DB, DB, T}) where {DA, DB, T}
    zero_AB = zeros(SMatrix{DA, DB, T})
    zero_BA = zeros(SMatrix{DB, DA, T})
    return [[A zero_AB]; [zero_BA B]]
end

function ChainRulesCore.rrule(::typeof(blk_diag), A::SMatrix{DA, DA, T}, B::SMatrix{DB, DB, T}) where {DA, DB, T}
    function blk_diag_adjoint(Δ)
        ΔA = Δ[SVector{DA}(1:DA), SVector{DA}(1:DA)]
        ΔB = Δ[SVector{DB}((DA+1):(DA+DB)), SVector{DB}((DA+1):(DA+DB))]
        return NoTangent(), ΔA, ΔB
    end
    return blk_diag(A, B), blk_diag_adjoint
end
