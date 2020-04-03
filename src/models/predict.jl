#
# Generic implementation. Good for StaticArrays.
#

@inline function predict(mf::AV, Pf::AM, A::AM, a::AV, Q::AM)
    return A * mf + a, (A * Pf) * A' + Q
end

@adjoint predict(m::AV, P::AM, A::AM, a::AV, Q::AM) = predict_pullback(m, P, A, a, Q)

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
# `A <: Matrix{<:Real}`.
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::AM{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))
    return predict!(mp, Pp, mf, Pf, A, a, Q)
end

function predict!(
    mp::Union{Vector{T}, SubArray{T, 1}},
    Pp::Union{Matrix{T}, SubArray{T, 2}},
    mf::Union{Vector{T}, SubArray{T, 1}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2}}},
    A::AM{T},
    a::Union{Vector{T}, SubArray{T, 1}},
    Q::Matrix{T},
) where {T<:Real}

    # Compute predictive mean.
    mp .= a
    mp = mul!(mp, A, mf, one(T), one(T))

    # Compute predictive covariance.
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))

    Pp .= Q
    Pp = mul!(Pp, APf, A', one(T), one(T))

    return mp, Pp
end

function predict_pullback(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::Matrix{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}

    # Pre-allocate for output.
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))

    # 1: Compute predictive mean.
    mp = mul!(copy!(mp, a), A, mf, one(T), one(T))

    # 2: compute A * Pf
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))

    # 3: compute APf * A' + Q
    Pp = mul!(copy!(Pp, Q), APf, A', one(T), one(T))

    return (mp, Pp), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # Pre-allocate for cotangents.
        Δmf = Vector{T}(undef, size(mf))
        ΔPf = Matrix{T}(undef, size(Pf))
        ΔAPf = Matrix{T}(undef, size(APf))
        ΔA = Matrix{T}(undef, size(A))

        # 3
        ΔQ = ΔPp
        ΔA = mul!(ΔA, ΔPp', APf)
        ΔAPf = mul!(ΔAPf, ΔPp, A)

        # 2
        ΔA = mul!(ΔA, ΔAPf, Pf', one(T), one(T))
        ΔPf = mul!(ΔPf, A', ΔAPf)

        # 1
        ΔA = mul!(ΔA, Δmp, mf', one(T), one(T))
        Δmf = mul!(Δmf, A', Δmp)
        Δa = Δmp

        return Δmf, ΔPf, ΔA, Δa, ΔQ
    end
end



#
# A <: BlockDiagonal{<:Real}
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))
    return predict!(mp, Pp, mf, Pf, A, a, Q)
end

function predict!(
    mp::Vector{T},
    Pp::Matrix{T},
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}

    # Compute predictive mean.
    mp = mul!(copy!(mp, a), A, mf, one(T), one(T))

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
            APf = getblock(A, n) * Pf.data[row_ids, col_ids]
            mul!(Pp[row_ids, col_ids], APf, getblock(A, m))
            col_lb += δ_r
        end

        # Shift the rows considered.
        row_lb += δ_r
    end
    return mp, Pp
end
