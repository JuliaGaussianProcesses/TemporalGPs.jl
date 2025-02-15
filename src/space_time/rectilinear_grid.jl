using Base.Iterators: product

"""
    RectilinearGrid{Tl, Tr} <: AbstractVector{Tuple{Tl, Tr}}

A `RectilinearGrid` is parametrised by `AbstractVector`s of points `xl` and `xr`, whose
element types are `Tl` and `Tr` respectively, comprising `length(xl) * length(xr)`
elements. Linear indexing is the same as `product(eachindex(xl), eachindex(xr))` - `xl`
iterates more quickly than `xr`.
"""
struct RectilinearGrid{Tl,Tr,Txl<:AbstractVector{Tl},Txr<:AbstractVector{Tr}} <:
       AbstractVector{Tuple{Tl,Tr}}
    xl::Txl
    xr::Txr
end

get_space(x::RectilinearGrid) = x.xl

get_times(x::RectilinearGrid) = x.xr

Base.size(X::RectilinearGrid) = (length(X.xl) * length(X.xr),)

function Base.collect(X::RectilinearGrid{Tl,Tr}) where {Tl,Tr}
    return vec(
        map(((p, q),) -> (X.xl[p], X.xr[q]), product(eachindex(X.xl), eachindex(X.xr)))
    )
end

function Base.getindex(X::RectilinearGrid, n::Integer)
    return (X.xl[mod(n - 1, length(X.xl)) + 1], X.xr[div(n - 1, length(X.xl)) + 1])
end

Base.show(io::IO, x::RectilinearGrid) = Base.show(io::IO, collect(x))

"""
    SpaceTimeGrid{Tr, Tt<:Real}

A `SpaceTimeGrid` is a `RectilinearGrid` in which the left vector corresponds to space, and
the right `time`. The left eltype is arbitrary, but the right must be `Real`.
"""
const SpaceTimeGrid{Tr,Tt<:Real} = RectilinearGrid{
    Tr,Tt,<:AbstractVector{Tr},<:AbstractVector{Tt}
}

#
# Implement internal API for transforming between "flat" representation, which is useful for
# GPs, an a time-centric representation, which is useful for state-space models.
#

# See docstring elsewhere for context.
function inputs_to_time_form(x::SpaceTimeGrid)
    return Fill(get_space(x), length(get_times(x)))
end

# See docstring elsewhere for context.
function merge_inputs(x1::SpaceTimeGrid, x2::SpaceTimeGrid)
    if get_space(x1) != get_space(x2)
        throw(error("Space coords of inputs not compatible, cannot merge."))
    end
    return RectilinearGrid(get_space(x1), vcat(get_times(x1), get_times(x2)))
end

# See docstring elsewhere for context.
function sort_in_time(x::SpaceTimeGrid)
    idx = sortperm(get_times(x))
    return idx, RectilinearGrid(get_space(x), get_times(x)[idx])
end

# See docstring elsewhere for context.
function observations_to_time_form(
    x::SpaceTimeGrid, y::AbstractVector{<:Union{Real,Missing}}
)
    return restructure(y, Fill(length(get_space(x)), length(get_times(x))))
end

function observations_to_time_form(x::SpaceTimeGrid, ::AbstractVector{Missing})
    return fill(missing, length(get_times(x)))
end

function get_zeros(x::SpaceTimeGrid{T}) where {T<:Real}
    return fill(Diagonal(zeros(T, length(get_space(x)))), length(get_times(x)))
end

# See docstring elsewhere for context.
function noise_var_to_time_form(x::RectilinearGrid, S::Diagonal{<:Real})
    vs = restructure(diag(S), Fill(length(get_space(x)), length(get_times(x))))
    return map(v -> Diagonal(collect(v)), vs)
end

destructure(::RectilinearGrid, y::AbstractVector) = reduce(vcat, y)
