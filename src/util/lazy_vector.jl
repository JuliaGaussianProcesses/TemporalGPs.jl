struct LazyVector{Tf, Tdata}
    f::Tf
    data::Tdata
end

Base.size(v::LazyVector) = size(v.data)

Base.getindex(v::LazyVector, n::Int) = v.f(data[n])

Base.map(g, v::LazyVector) = LazyVector(g ∘ v.f, v.data)

# Functionality directly supporting scan_emit's AD. It is checkpointed by default, because
# to do anything else would necessitate linear memory storage in the size of the array,
# which is something that we _really_ want to avoid.
function get_adjoint_storage(v::LazyVector, n::Int, Δxn)
    _, pb = Zygote._pullback(NoContext(), v.f, v.data[n])
    Δdata_n = pb(Δxn)[2]
    return (f=nothing, data=get_adjoint_storage(v.data, n, Δdata_n))
end

function _accum_at(Δv::NamedTuple{(:f, :data)}, n::Int, Δxn)
    _, pb = Zygote.pullback(NoContext(), v.f, v.data[n])
    Δdata_n = pb(Δxn)[2]
    return (f=nothing, data=_accum_at(Δv.data, n, Δdata_n))
end



# struct ZipArray{TAs<:Tuple}
#     As::TAs
# end

# Base.size(z::ZipArray) = size(first(As))

# Base.getindex(z::ZipArray, n::Int) = map(getindex)
