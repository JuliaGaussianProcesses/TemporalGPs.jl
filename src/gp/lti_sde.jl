"""
    LTISDE

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
    return vcat(map(marginals, marginals(build_lgssm(ft)))...)
end

function AbstractGPs.mean_and_var(ft::FiniteLTISDE)
    ms = marginals(ft)
    return map(mean, ms), map(var, ms)
end

AbstractGPs.mean(ft::FiniteLTISDE) = mean_and_var(ft)[1]

AbstractGPs.var(ft::FiniteLTISDE) = mean_and_var(ft)[2]

AbstractGPs.cov(ft::FiniteLTISDE) = cov(FiniteGP(ft.f.f, ft.x, ft.Σy))

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteLTISDE)
    return destructure(rand(rng, build_lgssm(ft)))
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
    model = build_lgssm(ft)
    return logpdf(model, restructure(y, model.emissions))
end

restructure(y::AbstractVector{<:Real}, ::StructArray{<:ScalarOutputLGC}) = y

destructure(y::AbstractVector{<:Real}) = y

# Converting GPs into LGSSMs.

function build_lgssm(ft::FiniteLTISDE)
    k = get_kernel(ft)
    x = Zygote.literal_getfield(ft, Val(:x))
    s = Zygote.literal_getfield(Zygote.literal_getfield(ft, Val(:f)), Val(:storage))
    As, as, Qs, emission_proj, x0 = lgssm_components(k, x, s)
    return LGSSM(
        GaussMarkovModel(Forward(), As, as, Qs, x0),
        build_emissions(emission_proj, build_Σs(ft)),
    )
end

function get_kernel(ft::FiniteLTISDE)
    return Zygote.literal_getfield(
        Zygote.literal_getfield(
            Zygote.literal_getfield(ft, Val(:f)), Val(:f),
        ),
        Val(:kernel),
    )
end

function build_Σs(ft::FiniteLTISDE)
    x = Zygote.literal_getfield(ft, Val(:x))
    Σy = Zygote.literal_getfield(ft, Val(:Σy))
    return build_Σs(x, Σy)
end

function build_Σs(::AbstractVector{<:Real}, Σ::Diagonal{<:Real})
    return Zygote.literal_getfield(Σ, Val(:diag))
end

function build_emissions(
    (Hs, hs)::Tuple{AbstractVector, AbstractVector}, Σs::AbstractVector,
)
    Hst = map(adjoint, Hs)
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
    k::SimpleKernel, t::AbstractVector, storage::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage)
    P = x0.P
    F, q, H = to_sde(k, storage)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    t = vcat([first(t) - 1], t)
    As = map(Δt -> time_exp(F, T(Δt)), diff(t))
    as = Fill(Zeros{T}(size(first(As), 1)), length(As))
    Qs = map(A -> Symmetric(P) - A * Symmetric(P) * A', As)
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
    F, q, H = to_sde(k, storage_type)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    A = time_exp(F, T(step(t)))
    As = Fill(A, length(t))
    as = Fill(Zeros{T}(size(F, 1)), length(t))
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

function to_sde(k::Matern12Kernel, s::SArrayStorage{T}) where {T<:Real}
    F = SMatrix{1, 1, T}(-1)
    q = convert(T, 2)
    H = SVector{1, T}(1)
    return F, q, H
end

function stationary_distribution(k::Matern12Kernel, s::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{1, T}(0),
        SMatrix{1, 1, T}(1),
    )
end



# Matern - 3/2

function to_sde(k::Matern32Kernel, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(3)
    F = SMatrix{2, 2, T}(0, -3, 1, -2λ)
    q = convert(T, 4 * λ^3)
    H = SVector{2, T}(1, 0)
    return F, q, H
end

function stationary_distribution(k::Matern32Kernel, ::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{2, T}(0, 0),
        SMatrix{2, 2, T}(1, 0, 0, 3),
    )
end



# Matern - 5/2

function to_sde(k::Matern52Kernel, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(5)
    F = SMatrix{3, 3, T}(0, 0, -λ^3, 1, 0, -3λ^2, 0, 1, -3λ)
    q = convert(T, 8 * λ^5 / 3)
    H = SVector{3, T}(1, 0, 0)
    return F, q, H
end

function stationary_distribution(k::Matern52Kernel, ::SArrayStorage{T}) where {T<:Real}
    κ = 5 / 3
    m = SVector{3, T}(0, 0, 0)
    P = SMatrix{3, 3, T}(1, 0, -κ, 0, κ, 0, -κ, 0, 25)
    return Gaussian(m, P)
end



# Constant

function TemporalGPs.to_sde(k::ConstantKernel, ::SArrayStorage{T}) where {T<:Real}
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

function _scale_emission_projections((Hs, hs)::Tuple{AbstractVector, AbstractVector}, σ)
    return (map(H->σ * H, Hs), map(h->σ * h, hs))
end

function _scale_emission_projections((Cs, cs, Hs, hs), σ)
    return (Cs, cs, map(H->σ * H, Hs), map(h->σ * h, hs))
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
    As_l, as_l, Qs_l, emission_proj_l, x0_l = lgssm_components(k.kernels[1], ts, storage_type)
    As_r, as_r, Qs_r, emission_proj_r, x0_r = lgssm_components(k.kernels[2], ts, storage_type)

    As = map(blk_diag, As_l, As_r)
    as = map(vcat, as_l, as_r)
    Qs = map(blk_diag, Qs_l, Qs_r)
    emission_projections = _sum_emission_projections(emission_proj_l, emission_proj_r)
    x0 = Gaussian(vcat(x0_l.m, x0_r.m), blk_diag(x0_l.P, x0_r.P))

    return As, as, Qs, emission_projections, x0
end

function _sum_emission_projections(
    (Hs_l, hs_l)::Tuple{AbstractVector, AbstractVector},
    (Hs_r, hs_r)::Tuple{AbstractVector, AbstractVector},
)
    return (map(vcat, Hs_l, Hs_r), hs_l + hs_r)
end

function _sum_emission_projections(
    (Cs_l, cs_l, Hs_l, hs_l)::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector},
    (Cs_r, cs_r, Hs_r, hs_r)::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector},
)
    Cs = map(vcat, Cs_l, Cs_r)
    cs = cs_l + cs_r
    Hs = map(blk_diag, Hs_l, Hs_r)
    hs = map(vcat, hs_l, hs_r)
    return (Cs, cs, Hs, hs)
end

Base.vcat(x::Zeros{T, 1}, y::Zeros{T, 1}) where {T} = Zeros{T}(length(x) + length(y))

function blk_diag(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    return hvcat(
        (2, 2),
        A, zeros(T, size(A, 1), size(B, 2)), zeros(T, size(B, 1), size(A, 2)), B,
    )
end

Zygote.@adjoint function blk_diag(A, B)
    function blk_diag_adjoint(Δ)
        ΔA = Δ[1:size(A, 1), 1:size(A, 2)]
        ΔB = Δ[size(A, 1)+1:end, size(A, 2)+1:end]
        return (ΔA, ΔB)
    end
    return blk_diag(A, B), blk_diag_adjoint
end

function blk_diag(A::SMatrix{DA, DA, T}, B::SMatrix{DB, DB, T}) where {DA, DB, T}
    zero_AB = zeros(SMatrix{DA, DB, T})
    zero_BA = zeros(SMatrix{DB, DA, T})
    return [[A zero_AB]; [zero_BA B]]
end

Zygote.@adjoint function blk_diag(
    A::SMatrix{DA, DA, T}, B::SMatrix{DB, DB, T},
) where {DA, DB, T}
    function blk_diag_adjoint(Δ::SMatrix)
        ΔA = Δ[SVector{DA}(1:DA), SVector{DA}(1:DA)]
        ΔB = Δ[SVector{DB}((DA+1):(DA+DB)), SVector{DB}((DA+1):(DA+DB))]
        return ΔA, ΔB
    end
    return blk_diag(A, B), blk_diag_adjoint
end
