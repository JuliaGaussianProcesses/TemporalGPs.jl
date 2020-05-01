abstract type StorageType{T<:Real} end

Base.eltype(::StorageType{T}) where {T<:Real} = T


struct SArrayStorage{T<:Real} <: StorageType{T} end

SArrayStorage(T) = SArrayStorage{T}()


struct ArrayStorage{T<:Real} <: StorageType{T} end

ArrayStorage(T) = ArrayStorage{T}()
