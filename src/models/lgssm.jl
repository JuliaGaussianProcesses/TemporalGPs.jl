export to_sde

abstract type AbstractSSM end

"""
    LGSSM <: AbstractSSM

A linear-Gaussian state-space model. Represented in terms of a Gauss-Markov model `gmm` and
a vector of observation covariance matrices.
"""
struct LGSSM{Tgmm<:GaussMarkovModel, TΣ<:AV{<:AM{<:Real}}} <: AbstractSSM
    gmm::Tgmm
    Σ::TΣ
end

Base.:(==)(x::LGSSM, y::LGSSM) = (x.gmm == y.gmm) && (x.Σ == y.Σ)

Base.length(ft::LGSSM) = length(ft.gmm)

dim_obs(ft::LGSSM) = dim_obs(ft.gmm)
dim_latent(ft::LGSSM) = dim_latent(ft.gmm)

function Base.getindex(model::LGSSM, n::Int)
    gmm = model.gmm
    return (A=gmm.A[n], a=gmm.a[n], Q=gmm.Q[n], H=gmm.H[n], h=gmm.h[n], Σ=model.Σ[n])
end

mean(model::LGSSM) = mean(model.gmm)

function cov(model::LGSSM)
    S = Stheno.cov(model.gmm)
    Σ = Stheno.block_diagonal(model.Σ)
    return S + Σ
end



#
# decorrelate
#

@inline function step_decorrelate(model, x::Gaussian, y::AV{<:Real})
    mp, Pp = _predict(x.m, x.P, model.A, model.a, model.Q)
    mf, Pf, lml, α = update_decorrelate(mp, Pp, model.H, model.h, model.Σ, y)
    return lml, α, Gaussian(mf, Pf)
end

@inline _predict(mf::AV, Pf::AM, A::AM, a::AV, Q::AM) = A * mf + a, (A * Pf) * A' + Q

@inline function update_decorrelate(mp, Pp, H::AM, h::AV, Σ::AM, y::AV{<:Real})
    V = H * Pp
    S_1 = V * H' + Σ
    S = cholesky(Symmetric(S_1))
    U = S.U
    B = U' \ V
    α = U' \ (y - H * mp - h)

    mf = mp + B'α
    Pf = Pp - B'B
    lml = -(length(y) * log(2π) + logdet(S) + α'α) / 2
    return mf, Pf, lml, α
end



#
# correlate
#

@inline function step_correlate(model, x::Gaussian, α::AV{<:Real})
    mp, Pp = _predict(x.m, x.P, model.A, model.a, model.Q)
    mf, Pf, lml, y = update_correlate(mp, Pp, model.H, model.h, model.Σ, α)
    return lml, y, Gaussian(mf, Pf)
end

@inline function update_correlate(mp, Pp, H::AM, h::AV, Σ::AM, α::AV{<:Real})
    V = H * Pp
    S = cholesky(Symmetric(V * H' + Σ))
    B = S.U' \ V
    y = S.U'α + H * mp + h

    mf = mp + B'α
    Pf = Pp - B'B
    lml = -(length(y) * log(2π) + logdet(S) + α'α) / 2
    return mf, Pf, lml, y
end

# Convert a latent Gaussian marginal into an observed Gaussian marginal.
to_observed(H::AM, h::AV, x::Gaussian) = Gaussian(H * x.m + h, H * x.P * H')


"""
    smooth(model::LGSSM, ys::AbstractVector)

Filter, smooth, and compute the log marginal likelihood of the data. Returns all
intermediate quantities.
"""
function smooth(model::LGSSM, ys::AbstractVector)

    lml, x_filter = filter(model, ys)

    # Smooth
    x_smooth = Vector{typeof(last(x_filter))}(undef, length(ys))
    x_smooth[end] = x_filter[end]
    for k in reverse(1:length(x_filter) - 1)
        x = x_filter[k]
        x′ = predict(model[k + 1], x_filter[k])

        U = cholesky(Symmetric(x′.P + 1e-12I)).U
        Gt = U \ (U' \ (model.gmm.A[k + 1] * x.P))
        x_smooth[k] = Gaussian(
            x.m + Gt' * (x_smooth[k + 1].m - x′.m),
            x.P + Gt' * (x_smooth[k + 1].P - x′.P) * Gt,
        )
    end

    Hs = model.gmm.H
    hs = model.gmm.h
    return to_observed.(Hs, hs, x_filter), to_observed.(Hs, hs, x_smooth), lml
end

predict(model, x) = Gaussian(_predict(x.m, x.P, model[1], model[2])...)

"""
    posterior_rand(rng::AbstractRNG, model::LGSSM, ys::Vector{<:AV{<:Real}})

Draw samples from the posterior over an LGSSM. This is not, currently, an especially
efficient implementation.
"""
function posterior_rand(
    rng::AbstractRNG,
    model::LGSSM,
    ys::Vector{<:AV{<:Real}},
    N_samples::Int,
)
    _, x_filter = filter(model, ys)

    chol_Q = cholesky.(Symmetric.(model.Q))

    x_T = rand(rng, x_filter[end], N_samples)
    x_sample = Vector{typeof(x_T)}(undef, length(ys))
    x_sample[end] = x_T
    for t in reverse(1:length(ys) - 1)

        # Produce joint samples.
        x̃ = rand(rng, x_filter[t], N_samples)
        x̃′ = model.A[t] * x̃ + chol_Q[t].U' * randn(rng, size(x_T)...)

        # Applying conditioning transformation.
        AP = model[t].A * x_filter[t].P
        S = Symmetric(model[t].A * Matrix(transpose(AP)) + model[t].Q)
        chol_S = cholesky(S)

        x_sample[t] = x̃ + AP' * (chol_S.U \ (chol_S.U' \ (x_sample[t+1] - x̃′)))
    end
    return model.H .* x_sample
end

function posterior_rand(rng::AbstractRNG, model::LGSSM, y::Vector{<:Real}, N_samples::Int)
    return posterior_rand(rng, model, [SVector{1}(yn) for yn in y], N_samples)
end



#
# High-level inference stuff that you really only want to have to write once...
#

pick_first(a, b) = a
get_pb(::typeof(pick_first)) = Δ->(Δ, nothing)

pick_last(a, b) = b
get_pb(::typeof(pick_last)) = Δ->(nothing, Δ)


for (foo, step_foo, step_foo_pullback) in [
    (:correlate, :step_correlate, :step_correlate_pullback),
    (:decorrelate, :step_decorrelate, :step_decorrelate_pullback),
]
    @eval function $foo(model::LGSSM, αs::AV{<:AV{<:Real}}, f=pick_first)
        @assert length(model) == length(αs)

        # Process first latent.
        lml, y, x = $step_foo(model[1], model.gmm.x0, first(αs))
        v = f(y, x)
        vs = Vector{typeof(v)}(undef, length(model))
        vs[1] = v

        # Process remaining latents.
        @inbounds for t in 2:length(model)
            lml_, y, x = $step_foo(model[t], x, αs[t])
            lml += lml_
            vs[t] = f(y, x)
        end
        return lml, vs
    end
end



#
# Things defined in terms of decorrelate
#

whiten(model::AbstractSSM, ys::AbstractVector) = last(decorrelate(model, ys))

Stheno.logpdf(model::AbstractSSM, ys::AbstractVector) = first(decorrelate(model, ys))

Base.filter(model::AbstractSSM, ys::AbstractVector) = decorrelate(model, ys, pick_last)

@adjoint function Base.filter(model::AbstractSSM, ys::AbstractVector)
    return Zygote.pullback(decorrelate, model, ys, pick_last)
end

# Resolve ambiguity with Base.
Base.filter(model::AbstractSSM, ys::Vector) = decorrelate(model, ys, pick_last)

@adjoint function Base.filter(model::AbstractSSM, ys::Vector)
    return Zygote.pullback(decorrelate, model, ys, pick_last)
end



#
# Things defined in terms of correlate
#

function Random.rand(rng::AbstractRNG, model::LGSSM)
    return last(correlate(model, rand_αs(rng, model, Val(dim_obs(model)))))
end

unwhiten(model::AbstractSSM, αs::AbstractVector) = last(correlate(model, αs))

function logpdf_and_rand(rng::AbstractRNG, model::LGSSM)
    return correlate(model, rand_αs(rng, model, Val(dim_obs(model))))
end

function rand_αs(rng::AbstractRNG, model::LGSSM, _)
    return [randn(rng, dim_obs(model)) for _ in 1:length(model)]
end

function rand_αs(
    rng::AbstractRNG,
    model::LGSSM{<:GaussMarkovModel{<:AV{<:SArray}}},
    ::Val{D},
) where {D}
    return [randn(rng, SVector{D}) for _ in 1:length(model)]
end
