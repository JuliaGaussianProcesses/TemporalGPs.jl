#
# This file contains pullbacks for stuff in immutable_inference.jl. There is no good reason
# to understand what's going on here.
#

#
# Objects in which to storage / accumulate the adjoint w.r.t. the hypers.
#

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

# Diagonal type constraint for the compiler's benefit.
@inline function _accum_at(Δxs::Vector{T}, n::Int, Δx::T) where {T}
    Δxs[n] = Δx
    return Δxs
end

# Diagonal type constraint for the compiler's benefit.
@inline function _accum_at(Δxs::NamedTuple{(:value,), Tuple{T}}, n::Int, Δx::T) where {T}
    return (value=Δxs.value + Δx,)
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

function get_pb(::typeof(copy_first))
    copy_first_pullback(Δ) = (copy(Δ), nothing)
    copy_first_pullback(Δ::Nothing) = (nothing, nothing)
    return copy_first_pullback
end

get_pb(::typeof(pick_last)) = Δ->(nothing, Δ)

for (foo, step_foo, foo_pullback) in [
    (:correlate, :step_correlate, :correlate_pullback),
    (:decorrelate, :step_decorrelate, :decorrelate_pullback),
]
    @eval @adjoint function $foo(model::LGSSM, ys::AV{<:AV{<:Real}}, f=copy_first)
        return $foo_pullback(model, ys, f)
    end

    # Standard rrule a la ChainRulesCore.
    @eval function $foo_pullback(model::LGSSM, ys::AV{<:AV{<:Real}}, f)
        @assert length(model) == length(ys)
        T = length(model)

        # Pre-allocate for filtering distributions. The indexing is slightly different for
        # these than for other quantities. In particular, xs[t] := x_{t-1}.
        xs = Vector{typeof(model.gmm.x0)}(undef, T + 1)
        xs[1] = model.gmm.x0 # the filtering distribution at t = 0

        # Process first observation.
        lml, α, x = $step_foo(model[1], xs[1], first(ys))
        xs[2] = x # the filtering distribution at t = 1

        # Allocate for remainder of operations.
        v = f(α, x)
        vs = Vector{typeof(v)}(undef, T)
        vs[1] = v

        # Process remaining observations.
        for t in 2:T
            lml_, α, x = $step_foo(model[t], xs[t], ys[t])
            xs[t + 1] = x # the filtering distribution at t = t.
            lml += lml_
            vs[t] = f(α, x)
        end

        # function foo_pullback(Δ::Tuple{Any, Nothing})
        #     return foo_pullback((Δ[1], Fill(nothing, T)))
        # end

        function foo_pullback(Δ::Tuple{Any, Union{AbstractVector, Nothing}})

            Δlml = Δ[1]
            Δvs = Δ[2] isa Nothing ? Fill(nothing, T) : Δ[2]

            # Compute the pullback through the last element of the chain to get
            # initialisations for cotangents to accumulate.
            Δys = Vector{eltype(ys)}(undef, T)
            (Δα, Δx__) = get_pb(f)(last(Δvs))
            _, pullback_last = _pullback(NoContext(), $step_foo, model[T], xs[T], ys[T])
            _, Δmodel_at_T, Δx, Δy = pullback_last((Δlml, Δα, Δx__))
            Δmodel = get_adjoint_storage(model, Δmodel_at_T)
            Δys[T] = Δy

            # Work backwards through the chain.
            for t in reverse(1:T-1)
                Δα, Δx__ = get_pb(f)(Δvs[t])
                Δx_ = Zygote.accum(Δx, Δx__)
                _, pullback_t = _pullback(NoContext(), $step_foo, model[t], xs[t], ys[t])
                _, Δmodel_at_t, Δx, Δy = pullback_t((Δlml, Δα, Δx_))
                Δmodel = _accum_at(Δmodel, t, Δmodel_at_t)
                Δys[t] = Δy
            end

            # Merge all gradient info associated with the model into the same place.
            Δmodel_ = (
                gmm = merge(Δmodel.gmm, (x0=Δx,)),
                Σ = Δmodel.Σ,
            )

            return nothing, Δmodel_, Δys, nothing
        end

        return (lml, vs), foo_pullback
    end
end
