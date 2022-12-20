# We force specialisation on all arguments to `scan_emit`, otherwise performance can drop
# off to an unacceptable degree when lots of compiler-intensive things like StaticArrays are
# used.
# See https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
# for more info.

"""
    Like Transducers.ScanEmit, but `f` isn't allowed to have internal state, and slightly
    faster in some cases that I care about for this package.

    This function makes some strong assumptions about the nature of `f` and `xs` so that it
    can achieve good performance. In particular, `f` must not involve any globals, and
    the output of `f` should not change type for different elements of `x`.
"""
function scan_emit(f, xs, state, idx)

    # Heuristic Warning: assume all ys have the same type as the 1st.
    (y, state) = f(state, _getindex(xs, idx[1]))
    ys = Vector{typeof(y)}(undef, length(idx))
    ys[idx[1]] = y

    for t in idx[2:end]
        (y, state) = f(state, _getindex(xs, t))
        ys[t] = y
    end

    return (ys, state)
end

function ChainRulesCore.rrule(::typeof(scan_emit), f, xs, init_state, idx)
    state = init_state
    (y, state) = f(state, _getindex(xs, idx[1]))

    # Heuristic Warning: assume all ys and states have the same type as the 1st.
    ys = Vector{typeof(y)}(undef, length(xs))
    states = Vector{typeof(state)}(undef, length(xs))

    ys[idx[1]] = y
    states[idx[1]] = state

    for t in idx[2:end]
        (y, state) = f(state, _getindex(xs, t))
        ys[t] = y
        states[t] = state
    end

    function scan_emit_pullback(Δ)

        Δ === nothing && return nothing
        Δys = Δ[1]
        Δstate = Δ[2]

        # This is a hack to handle the case that Δstate=nothing, and the "look at the
        # type of the first thing" heuristic breaks down.
        Δstate = Δ[2] === nothing ? _get_zero_adjoint(states[idx[end]]) : Δ[2]

        T = length(idx)
        if T > 1
            _, Δstate, Δx = step_pb(
                f, states[idx[T-1]], _getindex(xs, idx[T]), Δys[idx[T]], Δstate,
            )
            Δxs = get_adjoint_storage(xs, idx[T], Δx)

            for t in reverse(2:(T - 1))
                a = _getindex(xs, idx[t])
                b = Δys[idx[t]]
                c = states[idx[t-1]]
                _, Δstate, Δx = step_pb(
                    f, c, a, b, Δstate,
                )
                Δxs = _accum_at(Δxs, idx[t], Δx)
            end

            _, Δstate, Δx = step_pb(
                f, init_state, _getindex(xs, idx[1]), Δys[idx[1]], Δstate,
            )
            Δxs = _accum_at(Δxs, idx[1], Δx)

            return NoTangent(), NoTangent(), Δxs, Δstate, NoTangent()
        else
            _, Δstate, Δx = step_pb(
                f, init_state, _getindex(xs, idx[1]), Δys[idx[1]], Δstate,
            )
            Δxs = get_adjoint_storage(xs, idx[1], Δx)

            return NoTangent(), NoTangent(), Δxs, Δstate, NoTangent()
        end
    end

    return (ys, state), scan_emit_pullback
end

@inline function step_pb(f::Tf, state, x, Δy, Δstate) where {Tf}
    _, pb = _pullback(f, state, x)
    return pb((Δy, Δstate))
end

# Helper functionality for constructing appropriate differentials.

_getindex(x, idx::Int) = getindex(x, idx)

# This really ought to infer, but it doesn't for some unknown reason.
# _getindex(x::Base.Iterators.Zip, idx::Int) = map(x -> _getindex(x, idx), x.is)

# This is a work around for `map` not inferring properly.
_getindex(x::Base.Iterators.Zip, idx::Int) = __getindex(x.is, idx)
__getindex(x::Tuple{Any}, idx::Int) = (_getindex(x[1], idx), )
__getindex(x::Tuple, idx::Int) = (_getindex(x[1], idx), __getindex(Base.tail(x), idx)...)


_get_zero_adjoint(::Any) = nothing
_get_zero_adjoint(x::AbstractArray) = zero(x)



# Vector. In all probability, only one of these methods is necessary.

function get_adjoint_storage(x::Array, n::Int, Δx::T) where {T}
    x̄ = Array{T}(undef, size(x))
    x̄[n] = Δx
    return x̄
end

# function get_adjoint_storage(x::Vector{T}, n::Int, Δx::T) where {T<:Real}
#     x̄ = Vector{T}(undef, length(x))
#     x̄[n] = Δx
#     return x̄
# end

# function get_adjoint_storage(x::Vector, n::Int, init::T) where {T<:AbstractVecOrMat{<:Real}}
#     Δx = Vector{T}(undef, length(x))
#     Δx[n] = init
#     return Δx
# end

# function get_adjoint_storage(x::Vector, n::Int, init::T) where {T<:NamedTuple{(:diag,)}}
#     Δx = Vector{T}(undef, length(x))
#     Δx[n] = init
#     return Δx
# end



# Diagonal type constraint for the compiler's benefit.
@inline function _accum_at(Δxs::Vector{T}, n::Int, Δx::T) where {T}
    Δxs[n] = Δx
    return Δxs
end



# If there's nothing, there's nothing to do.

_accum_at(Δxs::Nothing, n::Int, Δx::Nothing) = nothing

# Zip

function get_adjoint_storage(x::Base.Iterators.Zip, n::Int, Δx::Tuple)
    return (is=map((x_, Δx_) -> get_adjoint_storage(x_, n, Δx_), x.is, Δx),)
end

# function _accum_at(Δxs::NamedTuple{(:is,)}, n::Int, Δx::Tuple)
#     return (is=map((Δxs_, Δx_) -> _accum_at(Δxs_, n, Δx_), Δxs.is, Δx), )
# end

# function _accum_at(Δxs::NamedTuple{(:is,)}, n::Int, Δx::Tuple{Any, Any})
#     return (is=(_accum_at(Δxs[1], n, Δx[1]), _accum_at(Δxs[2], n, Δx[2])), )
#     # return (is=map((Δxs_, Δx_) -> _accum_at(Δxs_, n, Δx_), Δxs.is, Δx), )
# end


# This is a work-around for `map` not inferring for some unknown reason. Very odd...
function _accum_at(Δxs::NamedTuple{(:is, )}, n::Int, Δx::Tuple)
    return (is=__accum_at(Δxs.is, n, Δx), )
end
__accum_at(Δxs::Tuple{Any}, n::Int, Δx::Tuple{Any}) = (_accum_at(Δxs[1], n, Δx[1]), )
function __accum_at(Δxs::Tuple, n::Int, Δx::Tuple)
    return (_accum_at(Δxs[1], n, Δx[1]), __accum_at(Base.tail(Δxs), n, Base.tail(Δx))...)
end


# Fill

get_adjoint_storage(::Fill, ::Int, init) = (value=init, axes=nothing)

# T is not parametrized since T can be SMatrix and Δx isa SizedMatrix
@inline function _accum_at(
    Δxs::NamedTuple{(:value, :axes), Tuple{T, Nothing}}, ::Int, Δx,
) where {T}
    return (value=accum(Δxs.value, Δx), axes=nothing)
end



# StructArray

function get_adjoint_storage(x::StructArray, n::Int, Δx::NamedTuple)
    init_arrays = map(
        (x_, Δx_) -> get_adjoint_storage(x_, n, Δx_), getfield(x, :components), Δx,
    )
    return (components = init_arrays, )
end

function get_adjoint_storage(x::StructArray, n::Int, Δx::StaticVector)
    init_arrays = map(
        (x_, Δx_) -> get_adjoint_storage(x_, n, Δx_), getfield(x, :components), Δx,
    )
    return (components = init_arrays, )
end

function _accum_at(Δxs::NamedTuple{(:components,)}, n::Int, Δx::NamedTuple)
    return (components = map((Δy, y) -> _accum_at(Δy, n, y), Δxs.components, Δx), )
end

function _accum_at(Δxs::NamedTuple{(:components,)}, n::Int, Δx::SVector)
    return (components = map((Δy, y) -> _accum_at(Δy, n, y), Δxs.components, Δx), )
end
