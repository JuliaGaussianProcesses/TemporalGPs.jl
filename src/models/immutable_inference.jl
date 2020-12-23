#
# High-level inference stuff that you really only want to have to write once...
#

Stheno.marginals(model::LGSSM) = scan_emit(step_marginals, model, model.gmm.x0)

function Stheno.logpdf(model::LGSSM, y::AbstractVector{<:AbstractVector{<:Real}})
    return sum(scan_emit(pick_lml ∘ step_decorrelate, zip(model, y), model.gmm.x0))
end

pick_lml(((lml, _), x)) = (lml, x)

function decorrelate(model::LGSSM, y::AbstractVector{<:AbstractVector{<:Real}})
    return scan_emit(pick_α ∘ step_decorrelate, zip(model, y), model.gmm.x0)
end

pick_α(((_, α), x)) = (α, x)

function _filter(model::LGSSM, y::AbstractVector{<:AbstractVector{<:Real}})
    return scan_emit(pick_x ∘ step_decorrelate, zip(model, y), model.gmm.x0)
end

pick_x((_, x)) = (x, x)

function correlate(model::LGSSM, α::AbstractVector{<:AbstractVector{<:Real}})
    return scan_emit(pick_y ∘ step_correlate, zip(model, α), model.gmm.x0)
end

pick_y(((_, y), x)) = (y, x)

Stheno.rand(rng::AbstractRNG, model::LGSSM) = correlate(model, rand_αs(rng, model))


# function Stheno.marginals(model::LGSSM)
    # # Allocate for marginals based on type of initial state.
    # x = predict(model[1], model.gmm.x0)
    # y = observe(model[1], x)
    # ys = Vector{typeof(y)}(undef, length(model))
    # ys[1] = y

    # # Process all latents.
    # for t in 2:length(model)
    #     x = predict(model[t], x)
    #     ys[t] = observe(model[t], x)
    # end

    # return ys
# end




# function decorrelate(model::LGSSM, ys::AbstractVector{T}) where {T<:AbstractVector{<:Real}}
#     @assert length(model) == length(ys)

#     x = model.gmm.x0
#     αs = Vector{T}(undef, length(model))
#     xs = Vector{typeof(x)}(undef, length(model))
#     lml = zero(eltype(model))

#     for t in 1:length(model)
#         lml_, α, x = step_decorrelate(model[t], x, ys[t])
#         lml += lml_
#         αs[t] = α
#         xs[t] = x
#     end

#     return lml, αs, xs
# end

# function correlate(model::LGSSM, αs::AbstractVector{T}) where {T<:AbstractVector{<:Real}}
#     @assert length(model) == length(αs)

#     x = model.gmm.x0
#     ys = Vector{T}(undef, length(model))
#     xs = Vector{typeof(x)}(undef, length(model))
#     lml = zero(eltype(model))

#     for t in 1:length(model)
#         lml_, y, x = step_correlate(model[t], x, αs[t])
#         lml += lml_
#         ys[t] = y
#         xs[t] = x
#     end
#     return lml, ys, xs
# end



#
# step
#

function step_marginals(x, model)
    x = predict(model, x)
    y = observe(model, x)
    return y, x
end

function step_decorrelate(x::Gaussian, (model, y)::Tuple{NamedTuple{(:gmm, :Σ)}, Any})
    gmm = model.gmm
    mp, Pp = predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)
    mf, Pf, lml, α = update_decorrelate(mp, Pp, gmm.H, gmm.h, model.Σ, y)
    return (lml, α), Gaussian(mf, Pf)
end

function step_correlate(x::Gaussian, (model, α)::Tuple{NamedTuple{(:gmm, :Σ)}, Any})
    gmm = model.gmm
    mp, Pp = predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)
    mf, Pf, lml, y = update_correlate(mp, Pp, gmm.H, gmm.h, model.Σ, α)
    return (lml, y), Gaussian(mf, Pf)
end



#
# predict and update
#

function predict(mf::AV{T}, Pf::AM{T}, A::AM{T}, a::AV{T}, Q::AM{T}) where {T<:Real}
    return A * mf + a, (A * Pf) * A' + Q
end

function update_decorrelate(
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

function update_correlate(
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
