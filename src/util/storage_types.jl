abstract type StorageType{T<:Real} end

Base.eltype(::StorageType{T}) where {T<:Real} = T



#
# Indicate to use `StaticArray`s to store model parameters.
#

struct SArrayStorage{T<:Real} <: StorageType{T} end

SArrayStorage(T) = SArrayStorage{T}()

mutability(::SArrayStorage) = Immutable()



#
# Indicate to use `Array`s to store model parameters.
#

struct ArrayStorage{T<:Real} <: StorageType{T} end

ArrayStorage(T) = ArrayStorage{T}()

mutability(::ArrayStorage) = Immutable()



#
# Is an array type to be considered Mutable or Immutable by this package?
#

struct Mutable end

struct Immutable end

Zygote.@nograd mutability
