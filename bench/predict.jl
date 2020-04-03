using BenchmarkTools, BlockDiagonals, LinearAlgebra, Random, TemporalGPs

using TemporalGPs: predict, predict_pullback, AV, AM



#
# Basline implementations to compare against.
#

naive_predict(mf, Pf, A, a, Q) = A * mf + a, (A * Pf) * A' + Q

function naive_predict_pullback(m::AV, P::AM, A::AM, a::AV, Q::AM)
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
# Dense * SymmetricDense * Dense + Dense
#

rng = MersenneTwister(123456);
D = 741;

# We don't need Pp or Q to be positive definite for any of the operations here.
A = randn(rng, D, D);
a = randn(rng, D);
Q = collect(Symmetric(randn(rng, D, D)));

mf = randn(rng, D);
Pf = Symmetric(randn(rng, D, D));

@benchmark naive_predict($mf, $Pf, $A, $a, $Q)
@benchmark predict($mf, $Pf, $A, $a, $Q)

@benchmark naive_predict_pullback($mf, $Pf, $A, $a, $Q)
@benchmark predict_pullback($mf, $Pf, $A, $a, $Q)

_, naive_dense_back = naive_predict_pullback(mf, Pf, A, a, Q);
_, dense_back = predict_pullback(mf, Pf, A, a, Q);

Δmp = randn(rng, D);
ΔPp = randn(rng, D, D);
@benchmark $naive_dense_back(($Δmp, $ΔPp))
@benchmark $dense_back(($Δmp, $ΔPp))



#
# BlockDiagonal * SymmetricDense * BlockDiagonal + BlockDiagonal
#

A = BlockDiagonal([randn(rng, D, D) for _ in 1:3]);
A_dense = collect(A);
a = randn(rng, size(A, 1));
Q = BlockDiagonal([randn(rng, D, D) for _ in 1:3]);
Q_dense = collect(Q);

mf = randn(rng, size(A, 1));
Pf = Symmetric(randn(rng, size(A)));

@benchmark naive_predict($mf, $Pf, $A_dense, $a, $Q_dense)
@benchmark predict($mf, $Pf, $A, $a, $Q)
