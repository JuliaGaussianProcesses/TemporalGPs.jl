#
# Generic implementation. Good for StaticArrays.
#

@inline function predict(mf::AV, Pf::AM, A::AM, a::AV, Q::AM)
    return A * mf + a, (A * Pf) * A' + Q
end

@adjoint function predict(m::AV, P::AM, A::AM, a::AV, Q::AM)
    return predict_pullback(m, P, A, a, Q)
end

function predict_pullback(m::AV, P::AM, A::AM, a::AV, Q::AM)
    mp = A * m + a # 1
    T = A * P # 2
    Pp = T * A' + Q # 3
    return (mp, Pp), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # 3
        ΔQ = ΔPp
        ΔA = ΔPp' * T
        ΔT = ΔPp * A

        # 2
        ΔA += ΔT * P'
        ΔP = A'ΔT

        # 1
        ΔA += Δmp * m'
        Δm = A'Δmp
        Δa = Δmp

        return Δm, ΔP, ΔA, Δa, ΔQ
    end
end



#
# Non-mutating wrapper interface for things that do mutation internally.
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::Union{Matrix{T}, BlockDiagonal{T}, KroneckerProduct{T}},
    a::Vector{T},
    Q::Union{Matrix{T}, BlockDiagonal{T}},
) where {T<:Real}
    mp = fill(zero(T), size(mf))
    Pp = fill(zero(T), size(Pf))
    return predict!(mp, Pp, mf, Pf, A, a, Q)
end

function predict_pullback(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::Union{Matrix{T}, BlockDiagonal{T}, KroneckerProduct{T, <:Eye{T}}},
    a::Vector{T},
    Q::Union{Matrix{T}, BlockDiagonal{T}},
) where {T<:Real}

    mp, Pp = predict(mf, Pf, A, a, Q)

    function predict_pullback_pullback(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # Pre-allocate for cotangents.
        Δmf = fill(zero(T), size(mf))
        ΔPf = fill(zero(T), size(Pf))
        ΔA = get_cotangent_storage(A, zero(T))
        Δa = fill(zero(T), size(a))
        ΔQ = get_cotangent_storage(Q, zero(T))

        return predict_pullback_accum!(Δmp, ΔPp, Δmf, ΔPf, ΔA, Δa, ΔQ, mf, Pf, A, a, Q)
    end

    return (mp, Pp), predict_pullback_pullback
end

get_cotangent_storage(A::Matrix{T}, val::T) where {T<:Real} = fill(val, size(A))

function get_cotangent_storage(A::BlockDiagonal{T}, val::T) where {T<:Real}
    return (blocks=map(block -> get_cotangent_storage(block, val), A.blocks), )
end

function get_cotangent_storage(
    A::KroneckerProduct{T, <:Eye, <:AbstractMatrix},
    val::T,
) where {T<:Real}
    return (A=nothing, B=get_cotangent_storage(A.B, val))
end



#
# `A <: Matrix{<:Real}`.
#

function predict!(
    mp::Union{Vector{T}, SubArray{T, 1}},
    Pp::Union{Matrix{T}, SubArray{T, 2}},
    mf::Union{Vector{T}, SubArray{T, 1}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2}}},
    A::Matrix{T},
    a::Union{Vector{T}, SubArray{T, 1}},
    Q::Matrix{T},
) where {T<:Real}

    # Compute predictive mean.
    mp .= a
    mp = mul!(mp, A, mf, one(T), one(T))

    # Compute predictive covariance.
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf)

    Pp .= Q
    Pp = mul!(Pp, APf, A', one(T), one(T))

    return mp, Pp
end

function predict_pullback_accum!(
    Δmp::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    ΔPp::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    Δmf::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    ΔPf::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    ΔA::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    Δa::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    ΔQ::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    mf::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2, Matrix{T}}}},
    A::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    a::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Q::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
) where {T<:Real}

    # Re-compute A * Pf
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf)

    # Pre-allocate for ΔAPf.
    ΔAPf = Matrix{T}(undef, size(APf))

    # 3
    ΔQ .+= ΔPp
    ΔA = mul!(ΔA, ΔPp', APf, one(T), one(T))
    ΔAPf = mul!(ΔAPf, ΔPp, A)

    # 2
    ΔA = mul!(ΔA, ΔAPf, Pf', one(T), one(T))
    ΔPf = mul!(ΔPf, A', ΔAPf, one(T), one(T))

    # 1
    ΔA = mul!(ΔA, Δmp, mf', one(T), one(T))
    Δmf = mul!(Δmf, A', Δmp, one(T), one(T))
    Δa .+= Δmp

    return Δmf, ΔPf, ΔA, Δa, ΔQ
end



#
# A <: BlockDiagonal{<:Real}
#

function predict!(
    mp::Vector{T},
    Pp::Matrix{T},
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, <:AbstractMatrix{T}},
    a::Vector{T},
    Q::BlockDiagonal{T, <:AbstractMatrix{T}},
) where {T<:Real}

    # Compute predictive covariance. Only works with the upper triangle.
    row_lb = 1
    @views for n in 1:nblocks(A)

        # Determine rows to consider.
        (δ_r, δ_c) = blocksize(A, n)
        @assert δ_r === δ_c
        row_ids = row_lb:(row_lb + δ_r - 1)

        # Update diagonal element of Pp.
        predict!(
            mp[row_ids],
            Pp[row_ids, row_ids],
            mf[row_ids],
            Symmetric(Pf.data[row_ids, row_ids]),
            getblock(A, n),
            a[row_ids],
            getblock(Q, n),
        )

        # Update elements above the diagonal.
        col_lb = row_lb + δ_r
        for m in (n + 1):nblocks(A)
            col_ids = col_lb:(col_lb + δ_r - 1)
            Al_Pf_Art!(
                Pp[row_ids, col_ids],
                getblock(A, n),
                Pf.data[row_ids, col_ids],
                getblock(A, m),
            )
            col_lb += δ_r
        end

        # Shift the rows considered.
        row_lb += δ_r
    end
    return mp, Pp
end

# Compute Al * Pf * Ar', storing the result in Pp. (l as in left, r as in right)
@inline function Al_Pf_Art!(
    Pp::SubArray{T, 2, Matrix{T}},
    Al::Matrix{T},
    Pf::SubArray{T, 2, Matrix{T}},
    Ar::Matrix{T},
) where {T<:Real}
    Al_Pf = Al * Pf
    mul!(Pp, Al_Pf, Ar')
    return nothing
end

function predict_pullback_accum!(
    Δmp::Vector{T},
    ΔPp::Matrix{T},
    Δmf::Vector{T},
    ΔPf::Matrix{T},
    ΔA::NamedTuple{(:blocks,)},
    Δa::Vector{T},
    ΔQ::NamedTuple{(:blocks,)},
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, <:AbstractMatrix{T}},
    a::Vector{T},
    Q::BlockDiagonal{T, <:AbstractMatrix{T}},
) where {T<:Real}

    # Compute predictive covariance. Only works with the upper triangle.
    row_lb = 1
    @views for n in 1:nblocks(A)

        # Determine rows to consider.
        (δ_r, δ_c) = blocksize(A, n)
        @assert δ_r === δ_c
        row_ids = row_lb:(row_lb + δ_r - 1)

        # Update diagonal element of Pp.
        predict_pullback_accum!(
            Δmp[row_ids],
            ΔPp[row_ids, row_ids],
            Δmf[row_ids],
            ΔPf[row_ids, row_ids],
            ΔA.blocks[n],
            Δa[row_ids],
            ΔQ.blocks[n],
            mf[row_ids],
            Symmetric(Pf.data[row_ids, row_ids]),
            getblock(A, n),
            a[row_ids],
            getblock(Q, n),
        )

        # Update elements above the diagonal.
        col_lb = row_lb + δ_r
        for m in (n + 1):nblocks(A)
            col_ids = col_lb:(col_lb + δ_r - 1)

            Al_Pf_Art_pullback!(
                ΔPp[row_ids, col_ids],
                ΔA.blocks[n],
                ΔPf[row_ids, col_ids],
                ΔA.blocks[m],
                getblock(A, n),
                Pf.data[row_ids, col_ids],
                getblock(A, m),
            )

            col_lb += δ_r
        end

        # Shift the rows considered.
        row_lb += δ_r
    end
    return Δmf, ΔPf, ΔA, Δa, ΔQ
end

# Compute Al * Pf * Ar', storing the result in Pp. (l as in left, r as in right)
@inline function Al_Pf_Art_pullback!(
    ΔPp::SubArray{T, 2, Matrix{T}},
    ΔAl::Matrix{T},
    ΔPf::SubArray{T, 2, Matrix{T}},
    ΔAr::Matrix{T},
    Al::Matrix{T},
    Pf::SubArray{T, 2, Matrix{T}},
    Ar::Matrix{T},
) where {T<:Real}

    # Recompute Al_Pf.
    Al_Pf = Al * Pf

    # Compute ΔAl_Pf and ΔAr.
    mul!(ΔAr, ΔPp', Al_Pf, one(T), one(T))
    ΔAl_Pf = ΔPp * Ar

    # Compute ΔAl and ΔPf.
    mul!(ΔAl, ΔAl_Pf, Pf', one(T), one(T))
    mul!(ΔPf, Al', ΔAl_Pf, one(T), one(T))
    return nothing
end



#
# A <: Kronecker{<:Real, <:Eye, <:Kronecker}
#

function predict!(
    mp::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Pp::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    mf::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2, Matrix{T}}}},
    A::KroneckerProduct{T, <:Eye{T}, Matrix{T}},
    a::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Q::Matrix{T},
) where {T<:Real}

    # Compute sizes.
    I_N, A_D = getmatrices(A)
    N = size(I_N, 1)
    D = size(A_D, 1)

    # Compute predictive mean, mp = A * mf + a.
    mp = copyto!(mp, a)
    mul!(reshape(mp, D, N), A_D, reshape(mf, D, N), one(T), one(T))

    # Compute APf = A * Pf.
    APf = Matrix{T}(undef, size(Pf))
    Pf = copy(Pf.data)
    LinearAlgebra.copytri!(Pf, 'U')
    mul!(reshape(APf, D, D * N^2), A_D, reshape(Pf, D, D * N^2))

    # Transpose APf.
    APft = Matrix{T}(undef, size(APf))
    @strided permutedims!(APft, APf, (2, 1))

    # Compute A * APft + Q.
    _compute_Pp!(Pp, A_D, APft, Q, D, N)

    return mp, Pp
end

function _compute_Pp!(
    Pp::Matrix{T},
    A_D::Matrix{T},
    APft::Matrix{T},
    Q::Matrix{T},
    D::Int,
    N::Int,
) where {T<:Real}
    Pp = copy!(Pp, Q)
    mul!(reshape(Pp, D, D * N^2), A_D, reshape(APft, D, D * N^2), one(T), one(T))
    return nothing
end

# Works with packed storage and copies the result into Pp at the end. This is necessary to
# achieve constant stride.
function _compute_Pp!(
    Pp::SubArray{T, 2, Matrix{T}},
    A_D::Matrix{T},
    APft::Matrix{T},
    Q::Matrix{T},
    D::Int,
    N::Int,
) where {T<:Real}
    Pp = copy!(Pp, Q)
    Pp_tmp = copy(Q)
    mul!(reshape(Pp_tmp, D, D * N^2), A_D, reshape(APft, D, D * N^2), one(T), one(T))
    copy!(Pp, Pp_tmp)
    return nothing
end

function predict_pullback_accum!(
    Δmp::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    ΔPp::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    Δmf::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    ΔPf::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    ΔA::NamedTuple{(:A, :B)},
    Δa::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    ΔQ::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
    mf::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2, Matrix{T}}}},
    A::KroneckerProduct{T, <:Eye{T}, Matrix{T}},
    a::Union{Vector{T}, SubArray{T, 1, Vector{T}}},
    Q::Union{Matrix{T}, SubArray{T, 2, Matrix{T}}},
) where {T<:Real}

    # Compute sizes.
    I_N, A_D = getmatrices(A)
    N = size(I_N, 1)
    D = size(A_D, 1)

    # Pull out cotangent storage for A_D.
    ΔA_D = ΔA.B

    #
    # Re-do some of the forwards-pass.
    #

    # Re-compute APf = A * Pf.
    APf = Matrix{T}(undef, size(Pf))
    Pf = Pf.data
    LinearAlgebra.copytri!(Pf, 'U')
    mul!(reshape(APf, D, D * N^2), A_D, reshape(Pf, D, D * N^2))

    # Transpose APf.
    APft = Matrix{T}(undef, size(APf))
    @strided permutedims!(APft, APf, (2, 1))

    #
    # Perform the reverse-pass.
    #

    # Compute cotangents arising from computing A * APft + Q.
    ΔQ .+= ΔPp
    mul!(ΔA_D, reshape(ΔPp, D, D * N^2), reshape(APft, D, D * N^2)', one(T), one(T))
    ΔAPft = Matrix{T}(undef, size(APft))
    mul!(reshape(ΔAPft, D, D * N^2), A_D', reshape(ΔPp, D, D * N^2))

    # Compute cotangents arising from computing transpose of APf.
    # Re-purposes memory allocated for APft to avoid the additional allocation.
    ΔAPf = APft
    @strided permutedims!(ΔAPf, ΔAPft, (2, 1))

    # Compute cotangets arising from computing APf = A * Pf.
    mul!(ΔA_D, reshape(ΔAPf, D, D * N^2), reshape(Pf, D, D * N^2)', one(T), one(T))
    mul!(reshape(ΔPf, D, D * N^2), A_D', reshape(ΔAPf, D, D * N^2), one(T), one(T))

    # Compute cotangents arising from computing mp = A * mf + a.
    Δa = copyto!(Δa, Δmp)
    mul!(ΔA_D, reshape(Δmp, D, N), reshape(mf, D, N)', one(T), one(T))
    mul!(reshape(Δmf, D, N), A_D', reshape(Δmp, D, N), one(T), one(T))

    return Δmf, ΔPf, ΔA, Δa, ΔQ
end

# Important for predict! with BlockDiagonals.
@inline function Al_Pf_Art!(
    Pp::SubArray{T, 2, Matrix{T}},
    Al::KroneckerProduct{T, <:Eye, Matrix{T}},
    Pf::SubArray{T, 2, Matrix{T}},
    Ar::KroneckerProduct{T, <:Eye, Matrix{T}},
) where {T<:Real}

    # Unpack the Kronecker matrices and determine sizes.
    Il, Al_D = getmatrices(Al)
    Ir, Ar_D = getmatrices(Ar)
    N = size(Il, 1)
    Dl = size(Al_D, 1)
    Dr = size(Ar_D, 1)

    # Compute Al_Pf = Al * Pf.
    Al_Pf = Matrix{T}(undef, size(Pf))
    mul!(reshape(Al_Pf, Dl, Dl * N^2), Al_D, reshape(Pf, Dl, Dl * N^2))

    # Transpose Al_Pf.
    Al_Pft = Matrix{T}(undef, size(Al_Pf))
    @strided permutedims!(Al_Pft, Al_Pf, (2, 1))

    # Compute Ar * Al_Pft to produce transpose(Pp).
    Ppt = Matrix{T}(undef, size(Pp))
    mul!(reshape(Ppt, Dr, Dr * N^2), Ar_D, reshape(Al_Pft, Dr, Dr * N^2))

    # Transpose Ppt to obtain Pp.
    @strided permutedims!(Pp, Ppt, (2, 1))

    return nothing
end

# Compute Al * Pf * Ar', storing the result in Pp. (l as in left, r as in right)
@inline function Al_Pf_Art_pullback!(
    ΔPp::SubArray{T, 2, Matrix{T}},
    ΔAl::NamedTuple{(:A, :B)},
    ΔPf::SubArray{T, 2, Matrix{T}},
    ΔAr::NamedTuple{(:A, :B)},
    Al::KroneckerProduct{T, <:Eye, Matrix{T}},
    Pf::SubArray{T, 2, Matrix{T}},
    Ar::KroneckerProduct{T, <:Eye, Matrix{T}},
) where {T<:Real}

    # Unpack the Kronecker matrices and determine sizes.
    Il, Al_D = getmatrices(Al)
    Ir, Ar_D = getmatrices(Ar)
    N = size(Il, 1)
    Dl = size(Al_D, 1)
    Dr = size(Ar_D, 1)
    ΔAl_D = ΔAl.B
    ΔAr_D = ΔAr.B

    # Recompute Al_Pf = Al * Pf.
    Al_Pf = Matrix{T}(undef, size(Pf))
    mul!(reshape(Al_Pf, Dl, Dl * N^2), Al_D, reshape(Pf, Dl, Dl * N^2))

    # Recompute the transpose of Al_Pf.
    Al_Pft = Matrix{T}(undef, size(Al_Pf))
    @strided permutedims!(Al_Pft, Al_Pf, (2, 1))

    # Compute ΔPpt from ΔPp.
    ΔPpt = Matrix{T}(undef, size(ΔPp))
    @strided permutedims!(ΔPpt, ΔPp, (2, 1))

    # Compute ΔAr_D and ΔAl_Pft. Primal op is Ppt = Ar * Al_Pft.
    mul!(ΔAr_D, reshape(ΔPpt, Dr, Dr * N^2), reshape(Al_Pft, Dr, Dr * N^2)', one(T), one(T))
    ΔAl_Pft = Al_Pft # Recycle memory for Al_Pft;.
    mul!(reshape(ΔAl_Pft, Dr, Dr * N^2), Ar_D', reshape(ΔPpt, Dr, Dr * N^2))

    # Transpose ΔAl_Pf.
    ΔAl_Pf = Al_Pf # recycle memory for Al_Pf
    @strided permutedims!(ΔAl_Pf, ΔAl_Pft, (2, 1))

    # Compute ΔAl and ΔPf. Primal op is Al_Pf = Al * Pf.
    mul!(ΔAl_D, reshape(ΔAl_Pf, Dl, Dl * N^2), reshape(Pf, Dl, Dl * N^2)', one(T), one(T))
    mul!(reshape(ΔPf, Dl, Dl * N^2), Al_D', reshape(ΔAl_Pf, Dl, Dl * N^2), one(T), one(T))

    return nothing
end
