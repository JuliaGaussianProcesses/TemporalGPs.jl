@inline symmetric(X::AbstractMatrix) = Symmetric(X)
@inline symmetric(X::Diagonal) = X

diag_Xt_invA_X(A::Cholesky, X::AbstractVecOrMat) = AbstractGPs.diag_At_A(A.U' \ X)

Xt_invA_X(A::Cholesky, x::AbstractVector) = sum(abs2, A.U' \ x)

function Xt_invA_X(A::Cholesky, X::AbstractMatrix)
    V = A.U' \ X
    return Symmetric(V'V)
end

function diag_At_B(A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B)
    return vec(sum(A .* B; dims=1))
end
