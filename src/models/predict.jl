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

    # Compute predictive mean.
    mp = mul!(copy(a), A, mf, 1.0, 1.0)

    # Compute predictive covariance.
    AQ = mul!(Matrix{T}(undef, size(Pf)), A, Pf, 1.0, 0.0)
    Pp = mul!(copy(Q), AQ, A', 1.0, 1.0)

    return mp, Pp
end

function predict!(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::AM{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}

    # Compute predictive mean.
    mp = mul!(copy(a), A, mf, 1.0, 1.0)

    # Compute predictive covariance.
    AQ = mul!(Matrix{T}(undef, size(Pf)), A, Pf, 1.0, 0.0)
    Pp = mul!(copy(Q), AQ, A', 1.0, 1.0)

    return mp, Pp
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
    return nothing
end
