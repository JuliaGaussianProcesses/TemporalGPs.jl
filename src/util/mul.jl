using BlockDiagonals: getblock

function LinearAlgebra.mul!(
    C::Matrix{T},
    A::BlockDiagonal{T},
    B::Matrix{T},
    α::T,
    β::T,
) where {T<:Real}
    @assert size(A, 1) == size(A, 2)
    start_row = 1

    @views for n in 1:nblocks(A)

        # Compute end row.
        (p, q) = blocksize(A, n)
        @assert p == q
        end_row = start_row + p - 1

        # Multiply nth block of A by rows of B.
        mul!(C[start_row:end_row, :], getblock(A, n), B[start_row:end_row, :], α, β)

        # Update position.
        start_row += p
    end
    return C
end


function LinearAlgebra.mul!(
    C::Matrix{T},
    A::BlockDiagonal{T},
    B::Adjoint{T, Matrix{T}},
    α::T,
    β::T,
) where {T<:Real}
    @assert size(A, 1) == size(A, 2)
    start_row = 1

    @views for n in 1:nblocks(A)

        # Compute end row.
        (p, q) = blocksize(A, n)
        @assert p == q
        end_row = start_row + p - 1

        # Multiply nth block of A by rows of B.
        mul!(C[start_row:end_row, :], getblock(A, n), B.parent[:, start_row:end_row]', α, β)

        # Update position.
        start_row += p
    end
    return C
end


function LinearAlgebra.mul!(
    C::Matrix{T},
    A::Matrix{T},
    B::BlockDiagonal{T},
    α::T,
    β::T,
) where {T<:Real}
    @assert size(B, 1) == size(B, 2)
    start_col = 1

    @views for n in 1:nblocks(B)

        # Compute end row.
        (p, q) = blocksize(B, n)
        @assert p == q
        end_col = start_col + p - 1

        # Multiply nth col-block A by nth block of B.
        mul!(C[:, start_col:end_col], A[:, start_col:end_col], getblock(B, n), α, β)

        # Update postion.
        start_col += p
    end
    return C
end

function LinearAlgebra.mul!(
    C::Matrix{T},
    A::Adjoint{T, Matrix{T}},
    B::BlockDiagonal{T},
    α::T,
    β::T,
) where {T<:Real}
    @assert size(B, 1) == size(B, 2)
    start_col = 1

    @views for n in 1:nblocks(B)

        # Compute end row.
        (p, q) = blocksize(B, n)
        @assert p == q
        end_col = start_col + p - 1

        # Multiply nth col-block A by nth block of B.
        mul!(C[:, start_col:end_col], A.parent[start_col:end_col, :]', getblock(B, n), α, β)

        # Update postion.
        start_col += p
    end
    return C
end
