#
# High-level inference stuff that you really only want to have to write once...
#

copy_first(a, b) = copy(a)

pick_last(a, b) = b

function Stheno.marginals(model::LGSSM)

    # Allocate for marginals based on type of initial state.
    x = predict(model[1], model.gmm.x0)
    y = observe(model[1], x)
    ys = Vector{typeof(y)}(undef, length(model))
    ys[1] = y

    # Process all latents.
    @inbounds for t in 2:length(model)
        x = predict(model[t], x)
        ys[t] = observe(model[t], x)
    end

    return ys
end

function decorrelate(model::LGSSM, ys::AV{<:AV{<:Real}}, f)
    @assert length(model) == length(ys)

    # Process first latent.
    lml, α, x = step_decorrelate(model[1], model.gmm.x0, first(ys))
    v = f(α, x)
    vs = Vector{typeof(v)}(undef, length(model))
    vs[1] = v

    # Process remaining latents.
    @inbounds for t in 2:length(model)
        lml_, α, x = step_decorrelate(model[t], x, ys[t])
        lml += lml_
        vs[t] = f(α, x)
    end
    return lml, vs
end

function correlate(model::LGSSM, αs::AV{<:AV{<:Real}}, f)
    @assert length(model) == length(αs)

    # Process first latent.
    lml, y, x = step_correlate(model[1], model.gmm.x0, first(αs))
    v = f(y, x)
    vs = Vector{typeof(v)}(undef, length(model))
    vs[1] = v

    # Process remaining latents.
    @inbounds for t in 2:length(model)
        lml_, y, x = step_correlate(model[t], x, αs[t])
        lml += lml_
        vs[t] = f(y, x)
    end
    return lml, vs
end



#
# step decorrelate / correlate
#

@inline function step_decorrelate(model::NamedTuple{(:gmm, :Σ)}, x::Gaussian, y::AV{<:Real})
    gmm = model.gmm
    mp, Pp = predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)
    mf, Pf, lml, α = update_decorrelate(mp, Pp, gmm.H, gmm.h, model.Σ, y)
    return lml, α, Gaussian(mf, Pf)
end

@inline function step_correlate(model::NamedTuple{(:gmm, :Σ)}, x::Gaussian, α::AV{<:Real})
    gmm = model.gmm
    mp, Pp = predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)
    mf, Pf, lml, y = update_correlate(mp, Pp, gmm.H, gmm.h, model.Σ, α)
    return lml, y, Gaussian(mf, Pf)
end



#
# predict and update
#

@inline function predict(mf::AV{T}, Pf::AM{T}, A::AM{T}, a::AV{T}, Q::AM{T}) where {T<:Real}
    return A * mf + a, (A * Pf) * A' + Q
end

@inline function update_decorrelate(
    mp::AV{T}, Pp::AM{T}, H::AM{T}, h::AV{T}, Σ::AM{T}, y::AV{T},
) where {T<:Real}

    V = H * Pp
    S_1 = V * H' + Σ
    S = cholesky(Symmetric(S_1))
    U = S.U
    B = U' \ V
    α = U' \ (y - H * mp - h)

    mf = mp + B'α
    Pf = _compute_Pf(Pp, B)
    lml = -(length(y) * T(log(2π)) + logdet(S) + α'α) / 2
    return mf, Pf, lml, α
end

@inline function update_correlate(
    mp::AV{T}, Pp::AM{T}, H::AM{T}, h::AV{T}, Σ::AM{T}, α::AV{T},
) where {T<:Real}

    V = H * Pp
    S = cholesky(Symmetric(V * H' + Σ))
    B = S.U' \ V
    y = S.U'α + (H * mp + h)

    mf = mp + B'α
    Pf = _compute_Pf(Pp, B)
    lml = -(length(y) * T(log(2π)) + logdet(S) + α'α) / 2
    return mf, Pf, lml, y
end

_compute_Pf(Pp::AM{T}, B::AM{T}) where {T<:Real} = Pp - B'B

# function _compute_Pf(Pp::Matrix{T}, B::Matrix{T}) where {T<:Real}
#     # Copy of Pp is necessary to ensure that the memory isn't modified.
#     # return BLAS.syrk!('U', 'T', -one(T), B, one(T), copy(Pp))
#     # I probably _do_ need a custom adjoint for this...
#     return LinearAlgebra.copytri!(BLAS.syrk!('U', 'T', -one(T), B, one(T), copy(Pp)), 'U')
# end
