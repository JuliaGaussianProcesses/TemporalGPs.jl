using BenchmarkTools, BlockDiagonals, LinearAlgebra, Random, TemporalGPs

using TemporalGPs: predict


naive_predict(mf, Pf, A, a, Q) = A * mf + a, (A * Pf) * A' + Q



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
