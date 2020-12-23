function posterior(prior::LGSSM, ys::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    Σs = Fill(small_noise_cov(eltype(prior.Σ), dim_obs(prior)), length(prior))
    return posterior(prior, ys, Σs)
end

function posterior(
    prior::LGSSM,
    ys::AbstractVector{<:AbstractVector{<:Real}},
    Σs_new::AbstractVector{<:AbstractMatrix},
)
    new_transition, xf = scan_emit(
        step_posterior_dynamics, zip(prior, ys), prior.gmm.x0, eachindex(prior),
    )
    A = map(x -> x.A, new_transition)
    a = map(x -> x.a, new_transition)
    Q = map(x -> x.Q, new_transition)
    H = prior.gmm.H
    h = prior.gmm.h
    return ReverseSSM(LGSSM(GaussMarkovModel(A, a, Q, H, h, xf), Σs_new))
end

function step_posterior_dynamics(
    xf::Gaussian, (prior, y)::Tuple{NamedTuple, AbstractVector{<:Real}},
)
    xp = predict(prior, xf)
    reverse_dynamics = invert_dynamics(xf, xp, prior)
    gmm = prior.gmm
    mf, Pf, _, _ = update_decorrelate(xp.m, xp.P, gmm.H, gmm.h, prior.Σ, y)
    return reverse_dynamics, Gaussian(mf, Pf)
end

function invert_dynamics(xf::Gaussian, xp::Gaussian, prior::NamedTuple)
    ε = convert(eltype(xf), 1e-12)
    U = cholesky(Symmetric(xp.P + UniformScaling(ε))).U
    Gt = U \ (U' \ (prior.gmm.A * xf.P))
    return (A=_collect(Gt'), a=xf.m - Gt'xp.m, Q=_compute_Pf(xf.P, U * Gt))
end

_collect(U::Adjoint{<:Any, <:Matrix}) = collect(U)
_collect(U::SMatrix) = U

small_noise_cov(::Type{<:SMatrix{D, D, T}}, ::Int) where {D, T} = SMatrix{D, D, T}(1e-12I)

small_noise_cov(::Type{Matrix{T}}, D::Int) where {T} = Matrix{T}(1e-12 * I, D, D)
