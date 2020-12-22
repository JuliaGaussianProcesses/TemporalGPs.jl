abstract type StorageType{T<:Real} end

Base.eltype(::StorageType{T}) where {T<:Real} = T

is_of_storage_type(::Any, ::StorageType) = false

is_of_storage_type(::T, ::StorageType{T}) where {T<:Real} = true

is_of_storage_type(x::AbstractArray{<:Real}, s::StorageType) = false

is_of_storage_type(x::AbstractArray, s::StorageType) = all(is_of_storage_type.(x, Ref(s)))

# A Tuple of objects are of a particular storage type if each element of the tuple is of
# that storage type.
function is_of_storage_type(xs::Union{Tuple, NamedTuple}, s::StorageType)
    return all(map(x -> is_of_storage_type(x, s), xs))
end

is_of_storage_type(x::Gaussian, s::StorageType) = is_of_storage_type((x.m, x.P), s)

is_of_storage_type(::Nothing, ::StorageType) = true


#
# Indicate to use `StaticArray`s to store model parameters.
#

struct SArrayStorage{T<:Real} <: StorageType{T} end

SArrayStorage(T) = SArrayStorage{T}()

is_of_storage_type(::SArray{<:Any, T}, ::SArrayStorage{T}) where {T<:Real} = true



#
# Indicate to use `Array`s to store model parameters.
#

struct ArrayStorage{T<:Real} <: StorageType{T} end

ArrayStorage(T) = ArrayStorage{T}()

is_of_storage_type(::Array{T}, ::ArrayStorage{T}) where {T<:Real} = true
