using Base.Iterators: product

"""
    RectilinearGrid{Tl, Tr} <: AV{Tuple{Tl, Tr}}

A `RectilinearGrid` is parametrised by `AbstractVector`s of points `xl` and `xr`, whose
element types are `Tl` and `Tr` respectively, comprising `length(xl) * length(xr)`
elements. Linear indexing is the same as `product(eachindex(xl), eachindex(xr))` - `xl`
iterates more quickly than `xr`.
"""
struct RectilinearGrid{Tl, Tr, Txl<:AV{Tl}, Txr<:AV{Tr}} <: AV{Tuple{Tl, Tr}}
    xl::Txl
    xr::Txr
end

Base.size(X::RectilinearGrid) = (length(X.xl) * length(X.xr),)

function Base.collect(X::RectilinearGrid{Tl, Tr}) where{Tl, Tr}
    return vec(
        map(
            ((p, q),) -> (X.xl[p], X.xr[q]),
            product(eachindex(X.xl), eachindex(X.xr)),
        )
    )
end

"""
    SpaceTimeGrid{Tr, Tt<:Real}

A `SpaceTimeGrid` is a `RectilinearGrid` in which the left vector corresponds to space, and
the right `time`. The left eltype is arbitrary, but the right must be `Real`.
"""
const SpaceTimeGrid{Tr, Tt<:Real} = RectilinearGrid{Tr, Tt, <:AV{Tr}, <:AV{Tt}}
