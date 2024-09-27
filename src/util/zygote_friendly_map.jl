"""
    zygote_friendly_map(f, x)

This version of map is a bit weird. It makes slightly stronger assumptions about the nature
of what you're allowed to pass in to it than `Base.map` does and, in return, you get much
improved performance when used in conjunction with `Zygote`.

# Assumptions.
- No globals are used in `f`. This means that `TemporalGPs.NoContext` can be employed.
- `f` has no fields. If you've got data to share across elements, use a `Fill`.
- Similarly, `f` has no mutable state (follows from the above).
- `f` doesn't mutate its argument.
"""
zygote_friendly_map(f, x) = dense_zygote_friendly_map(f, x)

function dense_zygote_friendly_map(f::Tf, x) where {Tf}

    # Perform first iteration.
    y_1 = f(_getindex(x, 1))

    # Allocate for outputs.
    ys = Array{typeof(y_1)}(undef, size(x))
    ys[1] = y_1

    # Perform remainder of iterations.
    for n in 2:length(x)
        ys[n] = f(_getindex(x, n))
    end

    return ys
end

zygote_friendly_map(f, x::Fill) = map(f, x)

function zygote_friendly_map(
    f, x::Base.Iterators.Zip{<:Tuple{Vararg{Fill, N}}},
) where {N}
    return zygote_friendly_map(f, Fill(map(first, x.is), length(x)))
end
