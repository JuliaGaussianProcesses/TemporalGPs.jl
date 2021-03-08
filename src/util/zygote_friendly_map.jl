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

function dense_zygote_friendly_map(f, x)

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

function Zygote._pullback(::AContext, ::typeof(dense_zygote_friendly_map), f, x)

    # Perform first iteration.
    y_1, pb_1 = Zygote._pullback(NoContext(), f, _getindex(x, 1))

    # Allocate for outputs.
    ys = Array{typeof(y_1)}(undef, size(x))
    ys[1] = y_1

    # Allocate for pullbacks.
    pbs = Array{typeof(pb_1)}(undef, size(x))
    pbs[1] = pb_1

    for n in 2:length(x)
        y, pb = Zygote._pullback(NoContext(), f, _getindex(x, n))
        ys[n] = y
        pbs[n] = pb
    end

    function zygote_friendly_map_pullback(Δ)
        Δ === nothing && return

        # Do first iteration.
        Δx_1 = pbs[1](Δ[1])

        # Allocate for cotangents.
        Δxs = get_adjoint_storage(x, 1, Δx_1[2])

        for n in 2:length(x)
            Δx = pbs[n](Δ[n])
            Δxs = _accum_at(Δxs, n, Δx[2])
        end

        return nothing, nothing, Δxs
    end

    return ys, zygote_friendly_map_pullback
end

zygote_friendly_map(f, x::Fill) = map(f, x)

function zygote_friendly_map(f, x::Base.Iterators.Zip{<:Tuple{Vararg{Fill, N}}}) where {N}
    return zygote_friendly_map(f, Fill(map(first, x.is), length(x)))
end
