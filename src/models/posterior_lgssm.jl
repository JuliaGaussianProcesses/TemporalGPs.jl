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

@inline function transition_dynamics(model::LGSSM)
    return map((A, a, Q) -> (A=A, a=a, Q=Q), model.gmm.A, model.gmm.a, model.gmm.Q)
end

function emission_dynamics(model::LGSSM)
    return map((H, h, Σ) -> (H=H, h=h, Σ=Σ), model.gmm.H, model.gmm.h, model.Σ)
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
    @show typeof(Σs_new)
    println("Doing a thing")
    prior = Zygote.hook(Δ -> ((@show typeof(Δ)); Δ), prior)
    p1 = Zygote.hook(Δ -> ((@show "p1", typeof(Δ)); Δ), prior)
    p2 = Zygote.hook(Δ -> ((@show "p2", typeof(Δ)); Δ), prior)
    p3 = Zygote.hook(Δ -> ((@show "p3", typeof(Δ)); Δ), prior)
    xfs = Zygote.hook(Δ -> ((@show typeof(Δ)); Δ), _filter(p1, y))
    transition = Zygote.hook(Δ -> ((@show typeof(Δ)); Δ), transition_dynamics(p2))
    new_dynamics = map(invert_dynamics, transition, emission_dynamics(p3), xfs, Σs_new)

    As = reverse(map(x -> x[1], new_dynamics))
    as = reverse(map(x -> x[2], new_dynamics))
    Qs = reverse(map(x -> x[3], new_dynamics))
    Hs = reverse(map(x -> x[4], new_dynamics))
    hs = reverse(map(x -> x[5], new_dynamics))
    Σs = reverse(map(x -> x[6], new_dynamics))

    # Create an arbitrary x0 that is consistent.
    t = transition[end]
    x_end = xfs[end]
    mp_, Pp_ = predict(x_end.m, x_end.P, t.A, t.a, t.Q)
    x0 = Gaussian(mp_, Pp_)

    return PosteriorLGSSM(LGSSM(GaussMarkovModel(As, as, Qs, Hs, hs, x0), Σs))
end

function posterior(
    prior::LGSSM,
    y::AbstractVector{<:AbstractVector{<:Real}},
    Σs_new::Fill,
)
    println("Doing this one")
    return posterior(prior, y, collect(Σs_new))
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
