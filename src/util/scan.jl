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

# Helper functionality for constructing appropriate differentials.

_getindex(x, idx::Int) = getindex(x, idx)

# This really ought to infer, but it doesn't for some unknown reason.
# _getindex(x::Base.Iterators.Zip, idx::Int) = map(x -> _getindex(x, idx), x.is)

# This is a work around for `map` not inferring properly.
_getindex(x::Base.Iterators.Zip, idx::Int) = __getindex(x.is, idx)
__getindex(x::Tuple{Any}, idx::Int) = (_getindex(x[1], idx),)
__getindex(x::Tuple, idx::Int) = (_getindex(x[1], idx), __getindex(Base.tail(x), idx)...)
