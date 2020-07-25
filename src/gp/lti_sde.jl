"""
    LTISDE

A lightweight wrapper around a `GP` `f` that tells this package to handle inference in `f`.
Can be constructed via the `to_sde` function. Indexing into this object produces a
`ScalarLGSSM`.
"""
struct LTISDE{Tf<:GP{<:Stheno.ZeroMean}, Tstorage<:StorageType} <: AbstractGP
    f::Tf
    storage::Tstorage
end

function to_sde(f::GP{<:Stheno.ZeroMean}, storage_type=ArrayStorage(Float64))
    return LTISDE(f, storage_type)
end
