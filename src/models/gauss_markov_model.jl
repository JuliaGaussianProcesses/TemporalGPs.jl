struct Forward end

struct Reverse end

"""
    GaussMarkovModel

Specifies the following Gauss-Markov model.
```julia
x[0] ∼ x0
x[t] = A[t] * x[t-1] + a[t] + ε[t], ε[t] ∼ N(0, Q)
```
"""
struct GaussMarkovModel{
    Tordering<:Union{Forward, Reverse},
    TAs<:AbstractVector{<:AbstractMatrix{<:Real}},
    Tas<:AbstractVector{<:AbstractVector{<:Real}},
    TQs<:AbstractVector{<:AbstractMatrix{<:Real}},
    Tx0<:Gaussian,
} <: AbstractMvNormal
    ordering::Tordering
    As::TAs
    as::Tas
    Qs::TQs
    x0::Tx0
end

function Zygote._pullback(::AContext, ::Type{<:GaussMarkovModel}, As, as, Qs, x0)
    function GaussMarkovModel_pullback(Δ)
        return (nothing, Δ.As, Δ.as, Δ.Qs, Δ.x0)
    end
    return GaussMarkovModel(As, as, Qs, x0), GaussMarkovModel_pullback
end

ordering(model::GaussMarkovModel) = model.ordering

Base.eltype(model::GaussMarkovModel) = eltype(first(model.As))

Base.eachindex(model::GaussMarkovModel{Forward}) = 1:length(model)

Base.eachindex(model::GaussMarkovModel{Reverse}) = reverse(1:length(model))

Base.length(model::GaussMarkovModel) = length(model.As)

function Base.getindex(model::GaussMarkovModel, n::Int)
    return LinearGaussianDynamics(model.As[n], model.as[n], model.Qs[n])
end

function Base.:(==)(x::GaussMarkovModel, y::GaussMarkovModel)
    return (x.As == y.As) && (x.as == y.as) && (x.Qs == y.Qs) && (x.x0 == y.x0)
end

dim(model::GaussMarkovModel) = length(first(model.as))

storage_type(model::GaussMarkovModel) = storage_type(first(model.As))

function is_of_storage_type(model::GaussMarkovModel, s::StorageType)
    return is_of_storage_type((model.As, model.as, model.Qs, model.x0), s)
end

x0(model::GaussMarkovModel) = model.x0

function get_adjoint_storage(x::GaussMarkovModel, Δx::NamedTuple{(:A, :a, :Q)})
    return (
        ordering = nothing,
        As = get_adjoint_storage(x.As, Δx.A),
        as = get_adjoint_storage(x.as, Δx.a),
        Qs = get_adjoint_storage(x.Qs, Δx.Q),
        x0 = nothing,
    )
end

function _accum_at(
    Δxs::NamedTuple{(:ordering, :As, :as, :Qs, :x0)},
    n::Int,
    Δx::NamedTuple{(:A, :a, :Q)},
)
    return (
        ordering = nothing,
        As = _accum_at(Δxs.As, n, Δx.A),
        as = _accum_at(Δxs.as, n, Δx.a),
        Qs = _accum_at(Δxs.Qs, n, Δx.Q),
        x0 = nothing,
    )
end