"""
    LTISDE

A lightweight wrapper around a `GP` `f` that tells this package to handle inference in `f`.
Can be constructed via the `to_sde` function. Indexing into this object produces a
`ScalarLGSSM`.
"""
struct LTISDE{Tf<:GP{<:Stheno.ZeroMean}, Tstorage<:StorageType}
    f::Tf
    storage::Tstorage
end

function to_sde(f::GP{<:Stheno.ZeroMean}, storage_type=ArrayStorage(Float64))
    return LTISDE(f, storage_type)
end

function (f::LTISDE)(t::AV{<:Real}, σ²s::AV{<:Real})
    model = LGSSM(GaussMarkovModel(f.f.k, t, f.storage), build_Σs(σ²s))
    return ScalarLGSSM(model)
end
(f::LTISDE)(t::AV, Σ::Diagonal{<:Real}) = f(t, Σ.diag)
(f::LTISDE)(t::AV, σ²::Real) = f(t, Fill(σ², length(t)))
(f::LTISDE)(t::AV) = f(t, zero(eltype(t)))



#
# This is all a bit ugly, and would ideally go. IIRC there's some issue with the
# interactions between `FillArrays` and `Zygote` here that is problematic.
#

build_Σs(σ²_ns::AbstractVector{<:Real}) = SMatrix{1, 1}.(σ²_ns)

@adjoint function build_Σs(σ²_ns::Vector{<:Real})
    function build_Σs_Vector_back(Δ)
        return (first.(Δ),)
    end
    return build_Σs(σ²_ns), build_Σs_Vector_back
end

@adjoint function build_Σs(σ²_ns::Fill{<:Real})
    function build_Σs_Fill_back(Δ::NamedTuple)
        return ((value=first(Δ.value),),)
    end
    return build_Σs(σ²_ns), build_Σs_Fill_back
end

# get_ms(m::MeanFunction, t::AV) = Stheno.ew(m, t)
# get_ms(m::ConstMean, t::AV) = Fill(m.c, length(t))
# get_ms(m::ZeroMean, t::AV) = Zeros(length(t))

# build_ms(ms::AbstractVector{<:Real}) = SVector{1}.(ms)

# @adjoint function build_ms(ms::Vector{<:Real})
#     function build_ms_Vector_back(Δ)
#         return (first.(Δ),)
#     end
#     return build_ms(ms), build_ms_Vector_back
# end

# @adjoint function build_ms(ms::Fill{<:Real})
#     function build_ms_Fill_back(Δ)
#         return ((value=first(Δ.value),),)
#     end
#     return build_ms(ms), build_ms_Fill_back
# end

# @adjoint build_ms(ms::Zeros{<:Real}) = SVector{1}.(ms), Δ->(nothing,)
