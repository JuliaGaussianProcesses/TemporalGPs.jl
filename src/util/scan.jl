using Zygote: AContext, _pullback

_getindex(x, idx::Int) = getindex(x, idx)
_getindex(x::Base.Iterators.Zip, idx::Int) = map(x -> _getindex(x, idx), x.is)

function get_adjoint_storage(x::Vector{T}, Δx::T) where {T<:Real}
    x̄ = Vector{T}(undef, length(x))
    x̄[end] = Δx
    return x̄
end

get_adjoint_storage(x::Base.Iterators.Zip, Δx::Tuple) = map(get_adjoint_storage, x.is, Δx)

function _accum_at(Δxs::Tuple, n::Int, Δx::Tuple)
    return map((Δxs_, Δx_) -> _accum_at(Δxs_, n, Δx_), Δxs, Δx)
end


"""
    Like Transducers.ScanEmit, but `f` isn't allowed to have internal state, and slightly
    faster in some cases that I care about for this package.
"""
function scan_emit(f, xs, state)

    # Heuristic Warning: assume all ys have the same type as the 1st.
    (y, state) = f(state, _getindex(xs, 1))
    ys = Vector{typeof(y)}(undef, length(xs))
    ys[1] = y

    for t in 2:length(xs)
        (y, state) = f(state, _getindex(xs, t))
        ys[t] = y
    end

    return ys
end

function Zygote._pullback(ctx::AContext, ::typeof(scan_emit), f, xs, init_state)

    state = init_state
    (y, state) = f(state, _getindex(xs, 1))

    # Heuristic Warning: assume all ys and states have the same type as the 1st.
    ys = Vector{typeof(y)}(undef, length(xs))
    states = Vector{typeof(state)}(undef, length(xs))

    ys[1] = y
    states[1] = state

    for t in 2:length(xs)
        (y, state) = f(state, _getindex(xs, t))
        ys[t] = y
        states[t] = state
    end

    step_pb(ctx, f, state, x, Δt, Δstate) = _pullback(ctx, f, state, x)[2]((Δt, Δstate))

    function scan_emit_pullback(Δ)

        Δ === nothing && return (nothing, nothing, nothing, nothing)

        T = length(xs)
        _, Δstate, Δx = step_pb(ctx, f, states[T-1], _getindex(xs, T), Δ[T], nothing)
        Δxs = get_adjoint_storage(xs, Δx)

        for t in reverse(2:(length(xs) - 1))
            _, Δstate, Δx = step_pb(ctx, f, states[t-1], _getindex(xs, t), Δ[t], Δstate)
            _accum_at(Δxs, t, Δx)
        end

        _, Δstate, Δx = step_pb(ctx, f, init_state, _getindex(xs, 1), Δ[1], Δstate)
        _accum_at(Δxs, 1, Δx)

        return (nothing, nothing, Δxs, Δstate)
    end

    return ys, scan_emit_pullback
end
