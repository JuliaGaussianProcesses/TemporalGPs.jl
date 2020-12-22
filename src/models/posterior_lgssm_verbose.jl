"""
    PosteriorLGSSM{Tmodel<:LGSSM, Txfs<:AbstractVector{<:Gaussian}} <: AbstractSSM

Represents the posterior distribution over an LGSSM given the filtering distributions xfs.
"""
struct PosteriorLGSSM{Tmodel<:LGSSM, Txfs<:AbstractVector{<:Gaussian}, TΣs} <: AbstractSSM
    model::Tmodel
    xfs::Txfs
    Σs::TΣs
end

function posterior(model::LGSSM, y::AbstractVector, Σs::AbstractVector{<:AbstractMatrix})
    return PosteriorLGSSM(model, _filter(model, y), Σs)
end

function posterior(model::LGSSM, y::AbstractVector)
    return PosteriorLGSSM(model, _filter(model, y), Fill(zeros(size(model.Σ[1])), length(model)))
end

# Base.:(==)(x::LGSSM, y::LGSSM) = (x.model == y.model) && (x.xfs == y.xfs)

Base.length(ft::PosteriorLGSSM) = length(ft.model)

# dim_obs(ft::LGSSM) = dim_obs(ft.model)

# dim_latent(ft::LGSSM) = dim_latent(ft.model)

# Base.eltype(ft::LGSSM) = eltype(ft.model)

# storage_type(ft::LGSSM) = storage_type(ft.model)

# Zygote.@nograd storage_type

function is_of_storage_type(model::PosteriorLGSSM, s::StorageType)
    return is_of_storage_type((model.model, model.xfs), s)
end

is_time_invariant(model::PosteriorLGSSM) = false

function transition_dynamics(model::LGSSM, t::Int)
    t = t > length(model) ? length(model) : t
    return (A=model.gmm.A[t], a=model.gmm.a[t], Q=model.gmm.Q[t])
end

transition_dynamics(model::PosteriorLGSSM, t::Int) = transition_dynamics(model.model, t)

function emission_dynamics(model::LGSSM, t::Int)
    return (H=model.gmm.H[t], h=model.gmm.h[t], Σ=model.Σ[t])
end

function emission_dynamics(model::PosteriorLGSSM, t::Int)
    return (H=model.model.gmm.H[t], h=model.model.gmm.h[t], Σ=model.Σs[t])
end


# mean(model::LGSSM) = mean(model.gmm)

# function cov(model::LGSSM)
#     S = Stheno.cov(model.gmm)
#     Σ = Stheno.block_diagonal(model.Σ)
#     return S + Σ
# end

_compute_ms(mf::AV, Gt::AM, ms′::AV, mp′::AV) = mf + Gt' * (ms′ - mp′)

_compute_Ps(Pf::AM, Gt::AM, Ps′::AM, Pp′::AM) = Pf + Gt' * (Ps′ - Pp′) * Gt

function _compute_Ps(
    Pf::Symmetric{<:Real, <:Matrix},
    Gt::Matrix,
    Ps′::Symmetric{<:Real, <:Matrix},
    Pp′::Matrix,
)
    return Symmetric(Pf + Gt' * (Ps′ - Pp′) * Gt)
end

# Pulls back xs a single step. Honestly, this is equivalent to predicting under the
# posterior model.
function step_smooth(
    xf::Gaussian, dynamics::NamedTuple{(:A, :a, :Q)}, xs::Gaussian,
) where {T<:Real}
    A, a, Q = dynamics
    mp, Pp = predict(xf.m, xf.P, A, a, Q)
    U = cholesky(Symmetric(Pp + convert(eltype(A), 1e-12) * I)).U
    Gt = U \ (U' \ (A * xf.P))
    return Gaussian(_compute_ms(xf.m, Gt, xs.m, mp), _compute_Ps(xf.P, Gt, xs.P, Pp)) 
end

# This is equivalent to smoothing.
function Stheno.marginals(model::PosteriorLGSSM)

    # Allocate for marginals based on the type of the initial state.
    T = length(model)
    x = model.xfs[T]
    y = observe(emission_dynamics(model, T)..., x)
    ys = Vector{typeof(y)}(undef, length(model))
    ys[end] = y

    for t in reverse(1:(length(model) - 1))
        x = step_smooth(model.xfs[t], transition_dynamics(model, t + 1), x)
        ys[t] = observe(emission_dynamics(model, t)..., x)
    end
    return ys
end

function decorrelate(model::PosteriorLGSSM, y::AbstractVector{<:AbstractVector{<:Real}})
    @assert length(model) == length(ys)

    αs = Vector{T}(undef, length(model))
    xs = Vector{typeof(x)}(undef, length(model))
    lml = zero(eltype(model))

    x = model.xfs[T]
    for t in reverse(1:(length(model) - 1))
        lml_, α, x = step_decorrelate_posterior(
            transition_dynamics(model, t + 1), emission_dynamics(model, t), x, ys[t],
        )
        lml += lml_
        αs[t] = α
        xs[t] = x
    end

    return lml, αs, xs
end

function correlate(model::PosteriorLGSSM, y::AbstractVector{<:AbstractVector{<:Real}})

end




"""
    smooth(model::LGSSM, ys::AbstractVector)
Filter, smooth, and compute the log marginal likelihood of the data. Returns all
intermediate quantities.
"""
function smooth(model::LGSSM, ys::AbstractVector)

    lml, _, x_filter = decorrelate(model, ys)
    ε = convert(eltype(model), 1e-12)

    # Smooth
    x_smooth = Vector{typeof(last(x_filter))}(undef, length(ys))
    x_smooth[end] = x_filter[end]
    for k in reverse(1:length(x_filter) - 1)
        x = x_filter[k]
        x′ = predict(model[k + 1], x)

        U = cholesky(Symmetric(x′.P + ε * I)).U
        Gt = U \ (U' \ (model.gmm.A[k + 1] * x.P))
        x_smooth[k] = Gaussian(
            _compute_ms(x.m, Gt, x_smooth[k + 1].m, x′.m),
            _compute_Ps(x.P, Gt, x_smooth[k + 1].P, x′.P),
        )
    end

    Hs = model.gmm.H
    hs = model.gmm.h
    return to_observed.(Hs, hs, x_filter), to_observed.(Hs, hs, x_smooth), lml
end

to_observed(H::AM, h::AV, x::Gaussian) = Gaussian(H * x.m + h, H * x.P * H')
