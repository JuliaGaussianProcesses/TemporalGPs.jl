"""
    decorrelate(::Mutable, model::LGSSM, ys::AbstractVector, f=copy_first)

Version of decorrelate used by `LGSSM`s whose `StorageType` is `Mutable`, as defined by
`mutability`.
"""
function decorrelate(::Mutable, model::LGSSM, ys::AbstractVector, f=copy_first)
    @assert length(model) == length(ys)

    # Pre-allocate for intermediates.
    α = Vector{eltype(first(ys))}(undef, length(first(ys)))
    x0 = model.gmm.x0
    x0_sym = Gaussian(x0.m, x0.P)
    mf = copy(x0.m)
    Pf = copy(x0.P)
    x = Gaussian(mf, Pf)

    # Process first latent.
    (lml, α, x) = step_decorrelate!(α, x, x0_sym, model[1], ys[1])
    v = f(α, x)
    vs = Vector{typeof(v)}(undef, length(model))
    vs[1] = v

    # Process remaining latents.
    @inbounds for t in 2:length(model)
        lml_, α, x = step_decorrelate!(α, x, x, model[t], ys[t])
        lml += lml_
        vs[t] = f(α, x)
    end
    return lml, vs
end

"""
    function step_decorrelate!(
        α::Vector{T},
        x_filter_next::Gaussian{Vector{T}, Matrix{T}},
        x_filter::Gaussian{Vector{T}, Matrix{T}},
        model::NamedTuple{(:gmm, :Σ)},
        y::Vector{<:Real},
    ) where {T<:Real}

Mutating version of `step_decorrelate`. Mutates both `α` and `x_filter_next`.
"""
function step_decorrelate!(
    α::Vector{T},
    x_filter_next::Gaussian{Vector{T}, Matrix{T}},
    x_filter::Gaussian{Vector{T}, Matrix{T}},
    model::NamedTuple{(:gmm, :Σ)},
    y::Vector{<:Real},
) where {T<:Real}

    # Preallocate for predictive distribution.
    mp = Vector{T}(undef, dim(x_filter))
    Pp = Matrix{T}(undef, dim(x_filter), dim(x_filter))
    x_predict = Gaussian(mp, Pp)

    # Compute next filtering distribution.
    gmm = model.gmm
    x_predict = predict!(x_predict, x_filter, gmm.A, gmm.a, gmm.Q)
    x_filter_next, lml, α = update_decorrelate!(
        α, x_filter_next, x_predict, gmm.H, gmm.h, model.Σ, y,
    )
    return lml, α, x_filter_next
end

"""
    predict!(
        x_predict::Gaussian{Vector{T}, Matrix{T}},
        x_filter::Gaussian{Vector{T}, Matrix{T}},
        A::Matrix{T},
        a::Vector{T},
        Q::Matrix{T},
    ) where {T<:Real}

Mutatiing version of `predict`. Modifies `x_predict`.
"""
function predict!(
    x_predict::Gaussian{Vector{T}, Matrix{T}},
    x_filter::Gaussian{Vector{T}, Matrix{T}},
    A::Matrix{T},
    a::AbstractVector{T},
    Q::Matrix{T},
) where {T<:Real}

    # Compute predictive mean.
    x_predict.m .= a
    mul!(x_predict.m, A, x_filter.m, one(T), one(T))

    # Compute predictive covariance.
    APf = Matrix{T}(undef, size(A))
    mul!(APf, A, x_filter.P)

    x_predict.P .= Q
    mul!(x_predict.P, APf, A', one(T), one(T))

    return x_predict
end

"""
    update_decorrelate!(
        α::Vector{T},
        x_filter::Gaussian{Vector{T}, Matrix{T}},
        x_predict::Gaussian{Vector{T}, Matrix{T}},
        H::Matrix{T},
        h::Vector{T},
        Σ::Matrix{T},
        y::AbstractVector{T},
    ) where {T<:Real}

Mutating version of `update_decorrelate`. Modifies `α` and `x_filter`.
"""
function update_decorrelate!(
    α::Vector{T},
    x_filter::Gaussian{Vector{T}, Matrix{T}},
    x_predict::Gaussian{Vector{T}, Matrix{T}},
    H::Matrix{T},
    h::AbstractVector{T},
    Σ::AbstractMatrix{T},
    y::AbstractVector{T},
) where {T<:Real}

    V = H * x_predict.P
    S_1 = V * H' + Σ
    S = cholesky(Symmetric(S_1))
    U = S.U
    B = U' \ V

    # α = U' \ (y - H * x_predict.m - h)
    α = ldiv!(α, U', y - H * x_predict.m - h)

    # x_filter.m .= x_predict.m + B'α
    x_filter.m .= x_predict.m
    mul!(x_filter.m, B', α, one(T), one(T))

    # x_filter.P .= x_predict.P - B'B
    x_filter.P .= x_predict.P
    _compute_Pf!(x_filter.P, B)

    # Compute log marginal probablilty of observation `y`.
    lml = -(length(y) * T(log(2π)) + logdet(S) + α'α) / 2

    return x_filter, lml, α
end

# Old method. Considering removing.
function _compute_Pf!(Pp::Matrix{T}, B::Matrix{T}) where {T<:Real}
    LinearAlgebra.copytri!(BLAS.syrk!('U', 'T', -1.0, B, 1.0, Pp), 'U')
end
