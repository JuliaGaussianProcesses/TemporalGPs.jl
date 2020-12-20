#
# This file contains pullbacks for stuff in immutable_inference.jl. There is no good reason
# to understand what's going on here unless you're working on AD and this package.
#

#
# Objects in which to storage / accumulate the adjoint w.r.t. the hypers.
#

zeroed_adjoint(x::Array{<:Real}) = zero(x)
zeroed_adjoint(x::SArray{<:Any, <:Real}) = zero(x)
zeroed_adjoint(::Type{T}) where {T<:Real} = zero(T)
zeroed_adjoint(::Type{Union{Nothing, T}}) where {T<:Real} = nothing

zeroed_adjoint(x::Gaussian) = (m=zeroed_adjoint(x.m), P=zeroed_adjoint(x.P))

get_adjoint(Δx::AbstractVector, n::Int) = Δx[n]
get_adjoint(::Nothing, ::Int) = nothing

get_adjoint(x, Δx::AbstractVector, n::Int) = Δx[n]
get_adjoint(x, ::Nothing, ::Int) = zeroed_adjoint(x)

# Diagonal type constraint for the compiler's benefit.
@inline function _accum_at(Δxs::Vector{T}, n::Int, Δx::T) where {T}
    Δxs[n] = Δx
    return Δxs
end

# Diagonal type constraint for the compiler's benefit.
@inline function _accum_at(Δxs::NamedTuple{(:value,), Tuple{T}}, n::Int, Δx::T) where {T}
    return (value=accum(Δxs.value, Δx),)
end

@inline function _accum_at(
    Δxs::NamedTuple{(:A, :a, :Q, :H, :h)},
    n::Int,
    Δx::NamedTuple{(:A, :a, :Q, :H, :h)},
)
    return (
        A = _accum_at(Δxs.A, n, Δx.A),
        a = _accum_at(Δxs.a, n, Δx.a),
        Q = _accum_at(Δxs.Q, n, Δx.Q),
        H = _accum_at(Δxs.H, n, Δx.H),
        h = _accum_at(Δxs.h, n, Δx.h),
    )
end

function _accum_at(Δxs::NamedTuple{(:gmm, :Σ)}, n::Int, Δx::NamedTuple{(:gmm, :Σ)})
    return (
        gmm = _accum_at(Δxs.gmm, n, Δx.gmm),
        Σ = _accum_at(Δxs.Σ, n, Δx.Σ),
    )
end

_accum_at(Δxs::Nothing, n::Int, Δx::Nothing) = nothing

function get_adjoint_storage(x::Vector, init::T) where {T<:AbstractVecOrMat{<:Real}}
    Δx = Vector{T}(undef, length(x))
    Δx[end] = init
    return Δx
end

get_adjoint_storage(x::Fill, init) = (value=init,)

function get_adjoint_storage(x::LGSSM, Δx::NamedTuple{(:gmm, :Σ)})
    return (
        gmm = get_adjoint_storage(x.gmm, Δx.gmm),
        Σ = get_adjoint_storage(x.Σ, Δx.Σ),
    )
end

function get_adjoint_storage(x::GaussMarkovModel, Δx::NamedTuple{(:A, :a, :Q, :H, :h)})
    return (
        A = get_adjoint_storage(x.A, Δx.A),
        a = get_adjoint_storage(x.a, Δx.a),
        Q = get_adjoint_storage(x.Q, Δx.Q),
        H = get_adjoint_storage(x.H, Δx.H),
        h = get_adjoint_storage(x.h, Δx.h),
    )
end

for (foo, step_foo, foo_pullback) in [
    (:correlate, :step_correlate, :correlate_pullback),
    (:decorrelate, :step_decorrelate, :decorrelate_pullback),
]
    @eval function Zygote._pullback(
        ::AContext, ::typeof($foo), model::LGSSM, ys::AbstractVector{<:AbstractVector},
    )
        lml, αs, xs = $foo(model, ys)

        function step_pullback(t::Int, Δlml, Δαs, Δx, model, ys)
            x = t > 1 ? xs[t - 1] : model.gmm.x0
            _, pb = _pullback(NoContext(), $step_foo, model[t], x, ys[t])
            Δ_t = (Δlml, get_adjoint(αs[t], Δαs, t), Δx)
            _, Δmodel_t, Δx, Δy = pb(Δ_t)
            return Δmodel_t, Δx, Δy
        end

        function $foo_pullback(
            Δ::Union{
                Nothing,
                Tuple{Any, Union{AbstractVector, Nothing}, Union{AbstractVector, Nothing}},
            },
        )

            Δ = Δ === nothing ? (nothing, nothing, nothing) : Δ
            Δlml = Δ[1]
            Δαs = Δ[2]
            Δxs = Δ[3]

            # Get model adjoint type by performing 1st iteration of reverse-pass.
            T = length(model)
            Δmodel_T, Δx, Δy = step_pullback(T, Δlml, Δαs, get_adjoint(Δxs, T), model, ys)
            Δmodel = get_adjoint_storage(model, Δmodel_T)
            Δys = get_adjoint_storage(ys, Δy)

            # Iterate backwards through the data.
            for t in reverse(1:(T-1))
                Δx = accum(get_adjoint(Δxs, t), Δx)
                Δmodel_t, Δx, Δy = step_pullback(t, Δlml, Δαs, Δx, model, ys)
                Δmodel = _accum_at(Δmodel, t, Δmodel_t)
                Δys[t] = Δy
            end

            return nothing, (gmm=merge(Δmodel.gmm, (x0=Δx,)), Σ=Δmodel.Σ), Δys
        end

        return (lml, αs, xs), $foo_pullback
    end

end
