#
# High-level inference stuff that you really only want to have to write once...
#

copy_first(a, b) = copy(a)

pick_last(a, b) = b

function decorrelate(::Immutable, model::LGSSM, ys::AV{<:AV{<:Real}}, f=copy_first)
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

function correlate(::Immutable, model::LGSSM, αs::AV{<:AV{<:Real}}, f=copy_first)
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

# # Immutable inference for heap-allocated arrays.
# @inline function predict(
#     mf::StridedVector{T},
#     Pf::StridedMatrix{T},
#     A::StridedMatrix{T},
#     a::StridedVector{T},
#     Q::StridedMatrix{T},
# ) where {T<:Real}

#     # Compute filtering mean vector.
#     mp = A * mf + a

#     # Compute filtering covariance matrix.
#     Pp = similar(Pf)
#     BLAS.copy!(Pp, Q)
#     mul!(Pp, A * Symmetric(Pf), A', one(T), one(T))

#     return mp, Pp
# end

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
    y = S.U'α + H * mp + h

    mf = mp + B'α
    Pf = _compute_Pf(Pp, B)
    lml = -(length(y) * T(log(2π)) + logdet(S) + α'α) / 2
    return mf, Pf, lml, y
end

_compute_Pf(Pp::AM{T}, B::AM{T}) where {T<:Real} = Pp - B'B

function _compute_Pf(Pp::Matrix{T}, B::Matrix{T}) where {T<:Real}
    # Copy of Pp is necessary to ensure that the memory isn't modified.
    # return BLAS.syrk!('U', 'T', -one(T), B, one(T), copy(Pp))
    return LinearAlgebra.copytri!(BLAS.syrk!('U', 'T', -one(T), B, one(T), copy(Pp)), 'U')
end
