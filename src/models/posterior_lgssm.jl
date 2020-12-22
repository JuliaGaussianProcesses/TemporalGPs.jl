struct PosteriorLGSSM{Tmodel<:LGSSM} <: AbstractSSM
    model::Tmodel
end

dim_obs(model::PosteriorLGSSM) = dim_obs(model.model)

Base.length(model::PosteriorLGSSM) = length(model.model)

dim_latent(ft::PosteriorLGSSM) = dim_latent(ft.model)

Base.eltype(ft::PosteriorLGSSM) = eltype(ft.model)

storage_type(ft::PosteriorLGSSM) = storage_type(ft.model)

is_of_storage_type(ft::PosteriorLGSSM, st::StorageType) = is_of_storage_type(ft.model, st)

function mean(model::PosteriorLGSSM)
    x = reverse(mean(model.model))
    D = dim_obs(model)
    x_unflipped = map(1:length(model)) do t
        reverse(view(x, ((t-1) * D + 1):(t * D)))
    end
    return vcat(x_unflipped...)
end

function cov(model::PosteriorLGSSM)
    x = reverse(reverse(cov(model.model); dims=1); dims=2)
    D = dim_obs(model)
    x_unflipped = map(1:length(model)) do t
        row_idx = ((t-1) * D + 1):(t * D)
        row_unflipped = map(1:length(model)) do v
            col_idx = ((v-1) * D + 1):(v * D)
            return reverse(view(x, row_idx, col_idx); dims=2)
        end
        return reverse(hcat(row_unflipped...); dims=1)
    end
    return vcat(x_unflipped...)
end

function transition_dynamics(model::LGSSM, t::Int)
    t = t > length(model) ? length(model) : t
    return (A=model.gmm.A[t], a=model.gmm.a[t], Q=model.gmm.Q[t])
end

function emission_dynamics(model::LGSSM, t::Int)
    return (H=model.gmm.H[t], h=model.gmm.h[t], Σ=model.Σ[t])
end

function invert_dynamics(transition, emission, xf::Gaussian, Σ_new::AbstractMatrix{<:Real})
    mp, Pp = predict(xf.m, xf.P, transition.A, transition.a, transition.Q)

    # Compute posterior transition dynamics.
    ε = convert(eltype(xf), 1e-12)
    U = cholesky(Symmetric(Pp + UniformScaling(ε))).U
    Gt = U \ (U' \ (transition.A * xf.P))
    A = _collect(Gt')
    a = xf.m - Gt'mp
    Q = _compute_Pf(xf.P, U * Gt)
    return (A, a, Q, emission.H, emission.h, Σ_new)
end

function posterior(
    prior::LGSSM,
    y::AbstractVector{<:AbstractVector{<:Real}},
    Σs_new::AbstractVector{<:AbstractMatrix},
)
    xfs = _filter(prior, y)
    transitions = map(t -> transition_dynamics(prior, t), eachindex(y) .+ 1)
    emissions = map(t -> emission_dynamics(prior, t), eachindex(y))
    new_dynamics = map(invert_dynamics, transitions, emissions, xfs, Σs_new)

    As = reverse(map(x -> x[1], new_dynamics))
    as = reverse(map(x -> x[2], new_dynamics))
    Qs = reverse(map(x -> x[3], new_dynamics))
    Hs = reverse(map(x -> x[4], new_dynamics))
    hs = reverse(map(x -> x[5], new_dynamics))
    Σs = reverse(map(x -> x[6], new_dynamics))

    # Create an arbitrary x0 that is consistent.
    transition = transition_dynamics(prior, length(prior) + 1)
    mp_, Pp_ = predict(xfs[end].m, xfs[end].P, transition.A, transition.a, transition.Q)
    x0 = Gaussian(mp_, Pp_)

    return PosteriorLGSSM(LGSSM(GaussMarkovModel(As, as, Qs, Hs, hs, x0), Σs))
end

_collect(U::Adjoint{<:Any, <:Matrix}) = collect(U)
_collect(U::SMatrix) = U

small_noise_cov(::Type{<:SMatrix{D, D, T}}, ::Int) where {D, T} = SMatrix{D, D, T}(1e-12I)

small_noise_cov(::Type{Matrix{T}}, D::Int) where {T} = Matrix{T}(1e-12 * I, D, D)

function posterior(prior::LGSSM, y::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    Σs = Fill(small_noise_cov(eltype(prior.Σ), dim_obs(prior)), length(prior))
    return posterior(prior, y, Σs)
end

Stheno.marginals(posterior::PosteriorLGSSM) = reverse(marginals(posterior.model))

function decorrelate(posterior::PosteriorLGSSM, y::AbstractVector)
    lml, α, xs = decorrelate(posterior.model, reverse(y))
    return lml, reverse(α), reverse(xs)
end

function correlate(posterior::PosteriorLGSSM, α::AbstractVector)
    lml, y, xs = correlate(posterior.model, reverse(α))
    return lml, reverse(y), reverse(xs)
end

rand_αs(rng::AbstractRNG, posterior::PosteriorLGSSM) = rand_αs(rng, posterior.model)




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
