using Stheno: ZeroMean, FiniteGP
using FillArrays: AbstractFill
export ssm, whiten

abstract type AbstractSSM end

"""
    LGSSM{TA, TQ, TH, TΣ, Tx₀} <: AbstractSSM

Classical linear Gaussian state-space model, with transition matrix `A`, transition-noise
variance `Q`, emission matrix `H` and observation-noise variance `Σ`. Also contains a number
of variables used during filtering to avoid additional allocations.
"""
struct LGSSM{TA<:AV{<:AM}, Tb<:AV{<:AV}, TQ<:AV{<:AM}, TH<:AV{<:AM}, Tc<:AV{<:AV}, TΣ<:AV{<:AM}, Tx₀} <: AbstractSSM
    A::TA
    b::Tb
    Q::TQ
    H::TH
    c::Tc
    Σ::TΣ
    x₀::Tx₀
end

function Base.getindex(model::LGSSM, n::Int)
    return (A=model.A[n], Q=model.Q[n], H=model.H[n], Σ=model.Σ[n])
end

function ==(x::LGSSM, y::LGSSM)
    return x.A == y.A && x.Q == y.Q && x.H == y.H &&
        x.Σ == y.Σ && x.x₀ == y.x₀
end

Base.length(model::LGSSM) = length(model.A)

dim_latent(model::LGSSM) = length(model.x₀.m)

dim_obs(model::LGSSM) = size(first(model.Σ), 1)

is_lti(::LGSSM) = false

is_lti(::LGSSM{
    <:AbstractFill, <:AbstractFill, <:AbstractFill, 
    <:AbstractFill, <:AbstractFill, <:AbstractFill,
}) = true



#
# Construct LGSSMs from LTISDEs.
#

"""
    ssm(sde::LTISDE, t::StepRangeLen, Σs::AV{<:AM})

Construct a time-invariant LGSSM from `sde` with time-step `Δt` and observation noise
covariance matrices `Σs`.
"""
function ssm(sde::LTISDE, t::StepRangeLen, cs::AV{<:AV}, Σs::AV{<:AM})
    A = time_exp(sde.F, sde.v * step(t))
    Q = sde.x₀.P - A * sde.x₀.P * A'

    As = Fill(A, length(t))
    bs = Fill(Zeros(size(A, 1)), length(t))
    Qs = Fill(Q, length(t))
    Hs = Fill(sde.H, length(t))

    return LGSSM(As, bs, Qs, Hs, cs, Σs, sde.x₀)
end

"""
    ssm(sde::LTISDE, t::AbstractVector{<:Real}, Σs::AV{<:AM})

Construct an LGSSM from `sde` at time-point `t` with observation noise covariance matrices
`Σs`.
"""
function ssm(sde::LTISDE, t::AV{<:Real}, cs::AV{<:AV}, Σs::AV{<:AM})
    t = vcat([first(t) - 1], t)
    As = map(Δt -> time_exp(sde.F, sde.v * Δt), diff(t))
    bs = Fill(Zeros(size(first(As), 1)), length(t))
    P = sde.x₀.P
    Qs = map(A -> P - A * P * A', As)
    Hs = Fill(sde.H, length(As))
    return LGSSM(As, bs, Qs, Hs, cs, Σs, sde.x₀)
end



#
# decorrelate
#

@inline function step_decorrelate(model, x::Gaussian, y::AV{<:Real})
    mp, Pp = _predict(x.m, x.P, model.A, model.Q)
    mf, Pf, lml, α = update_decorrelate(mp, Pp, model.H, model.Σ, y)
    return lml, α, Gaussian(mf, Pf)
end

@inline _predict(mf::AV, Pf::AM, A::AM, Q::AM) = A * mf, (A * Pf) * A' + Q

@inline function update_decorrelate(mp, Pp, H::AM, Σ::AM, y::AV{<:Real})
    V = H * Pp
    S_1 = V * H' + Σ
    S = cholesky(Symmetric(S_1))
    U = S.U
    B = U' \ V
    α = U' \ (y - H * mp)

    mf = mp + B'α
    Pf = Pp - B'B
    lml = -(length(y) * log(2π) + logdet(S) + α'α) / 2
    return mf, Pf, lml, α
end



#
# correlate
#

@inline function step_correlate(model, x::Gaussian, α::AV{<:Real})
    mp, Pp = _predict(x.m, x.P, model.A, model.Q)
    mf, Pf, lml, y = update_correlate(mp, Pp, model.H, model.Σ, α)
    return lml, y, Gaussian(mf, Pf)
end

@inline function update_correlate(mp, Pp, H::AM, Σ::AM, α::AV{<:Real})
    V = H * Pp
    S = cholesky(Symmetric(V * H' + Σ))
    B = S.U' \ V
    y = S.U'α + H * mp

    mf = mp + B'α
    Pf = Pp - B'B
    lml = -(length(y) * log(2π) + logdet(S) + α'α) / 2
    return mf, Pf, lml, y
end

# Convert a latent Gaussian marginal into an observed Gaussian marginal.
to_observed(H::AM, x::Gaussian) = Gaussian(H * x.m, H * x.P * H')


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
        Gt = U \ (U' \ (model.A[k + 1] * x.P))
        x_smooth[k] = Gaussian(
            x.m + Gt' * (x_smooth[k + 1].m - x′.m),
            x.P + Gt' * (x_smooth[k + 1].P - x′.P) * Gt,
        )
    end

    return to_observed.(model.H, x_filter), to_observed.(model.H, x_smooth), lml
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
        lml, y, x = $step_foo(model[1], model.x₀, first(αs))
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

    # Standard rrule a la ChainRulesCore.
    @eval @adjoint function $foo(model::LGSSM, ys::AV{<:AV{<:Real}}, f=pick_first)

        @assert length(model) == length(ys)

        # Process first observation.
        (lml, α, x), pb = $step_foo_pullback(model[1], model.x₀, first(ys))
        v = f(α, x)

        # Allocate for remainder of operation.
        vs = Vector{typeof(v)}(undef, length(model))
        pullbacks = Vector{typeof(pb)}(undef, length(ys))
        vs[1] = v
        pullbacks[1] = pb

        # Process remaining observations.
        for t in 2:length(ys)
            (lml_, α, x), pb = $step_foo_pullback(model[t], x, ys[t])
            lml += lml_
            vs[t] = f(α, x)
            pullbacks[t] = pb
        end

        return (lml, vs), function(Δ)

            Δlml = Δ[1]
            Δvs = Δ[2] === nothing ? Fill(nothing, length(vs)) : Δ[2]

            Δys = Vector{eltype(ys)}(undef, length(ys))
            (Δα, Δx) = get_pb(f)(last(Δvs))
            Δmodel_1, Δx, Δy = last(pullbacks)((Δlml, Δα, Δx))

            Δmodel = (
                A = get_adjoint_storage(model.A, Δmodel_1.A),
                Q = get_adjoint_storage(model.Q, Δmodel_1.Q),
                H = get_adjoint_storage(model.H, Δmodel_1.H),
                Σ = get_adjoint_storage(model.Σ, Δmodel_1.Σ),
                x₀ = nothing,
            )
            Δys[end] = Δy

            for t in reverse(1:length(ys)-1)
                Δα, Δx_ = get_pb(f)(Δvs[t])
                Δx = Zygote.accum(Δx, Δx_)
                Δmodel_, Δx, Δy = pullbacks[t]((Δlml, Δα, Δx))
                Δmodel = _accum_at(Δmodel, t, Δmodel_)
                Δys[t] = Δy
            end
            Δmodel = (
                A = Δmodel.A,
                b = nothing,
                Q = Δmodel.Q,
                H = Δmodel.H,
                c = nothing,
                Σ = Δmodel.Σ,
                x₀ = nothing,
            )
            return Δmodel, Δys, nothing
        end
    end
end

function get_adjoint_storage(x::Vector, init::T) where {T<:AbstractVecOrMat{<:Real}}
    Δx = Vector{T}(undef, length(x))
    Δx[end] = init
    return Δx
end
get_adjoint_storage(x::Fill, init) = (value=init,)

function _accum_at(Δxs::Vector, n::Int, Δx)
    Δxs[n] = Δx
    return Δxs
end

_accum_at(Δxs::NamedTuple{(:value,)}, n::Int, Δx) = (value=Δxs.value + Δx,)

function _accum_at(Δxs::NamedTuple{(:A, :Q, :H, :Σ, :x₀)}, n::Int, Δx)
    return (
        A = _accum_at(Δxs.A, n, Δx.A),
        Q = _accum_at(Δxs.Q, n, Δx.Q),
        H = _accum_at(Δxs.H, n, Δx.H),
        Σ = _accum_at(Δxs.Σ, n, Δx.Σ),
        x₀ = nothing,
    )
end

#
# Things defined in terms of decorrelate
#

whiten(model::AbstractSSM, ys::AbstractVector) = last(decorrelate(model, ys))

Stheno.logpdf(model::AbstractSSM, ys::AbstractVector) = first(decorrelate(model, ys))

Base.filter(model::AbstractSSM, ys::AbstractVector) = decorrelate(model, ys, pick_last)

@adjoint function filter(model::AbstractSSM, ys::AbstractVector)
    return Zygote.pullback(decorrelate, model, ys, pick_last)
end

# Resolve ambiguity with Base.
Base.filter(model::AbstractSSM, ys::Vector) = decorrelate(model, ys, pick_last)

@adjoint function filter(model::AbstractSSM, ys::Vector)
    return Zygote.pullback(decorrelate, model, ys, pick_last)
end


#
# Things defined in terms of correlate
#

function Random.rand(rng::AbstractRNG, model::LGSSM)
    return last(correlate(model, [randn(rng, dim_obs(model)) for _ in 1:length(model)]))
end

function Random.rand(rng::AbstractRNG, model::LGSSM{<:AV{<:StaticMatrix}})
    αs_real = randn(rng, length(model))
    αs = reinterpret(SVector{1, eltype(αs_real)}, αs_real)
    return last(correlate(model, αs))
end

function unwhiten(model::AbstractSSM, αs::AbstractVector)
    return first(correlate(model, αs))
end

function logpdf_and_rand(rng::AbstractRNG, model::LGSSM)
    return correlate(model, [randn(rng, dim_obs(model)) for _ in 1:length(model)])
end
