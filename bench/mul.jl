using BenchmarkTools, BlockDiagonals, LinearAlgebra, Random, TemporalGPs

rng = MersenneTwister(123456);
P = 50;
Q = 150;

# Matrix-Matrix multiplies.

A = BlockDiagonal([randn(rng, P, P) for _ in 1:3]);
A_dense = collect(A);
At = BlockDiagonal(map(collect ∘ transpose, blocks(A)));
At_dense = collect(At);
B = randn(rng, Q, Q);
C = randn(rng, Q, Q);

@benchmark mul!($C, $A, $B, 1.0, 1.0)
@benchmark mul!($C, $A_dense, $B, 1.0, 1.0)

@benchmark mul!($C, $At', $B, 1.0, 1.0)
@benchmark mul!($C, $At_dense', $B, 1.0, 1.0)

@benchmark mul!($C, $A, $B', 1.0, 1.0)
@benchmark mul!($C, $A_dense, $B', 1.0, 1.0)

@benchmark mul!($C, $At', $B', 1.0, 1.0)
@benchmark mul!($C, $At_dense', $B', 1.0, 1.0)

@benchmark mul!($C, $B, $A, 1.0, 1.0)
@benchmark mul!($C, $B, $A_dense, 1.0, 1.0)

@benchmark mul!($C, $B, $At', 1.0, 1.0)
@benchmark mul!($C, $B, $At_dense', 1.0, 1.0)

@benchmark mul!($C, $B', $A, 1.0, 1.0)
@benchmark mul!($C, $B', $A_dense, 1.0, 1.0)

@benchmark mul!($C, $B', $At', 1.0, 1.0)
@benchmark mul!($C, $B', $At_dense', 1.0, 1.0)

# Matrix-Vector multiplies.

b = randn(rng, Q);
c = randn(rng, Q);

@benchmark mul!($c, $A, $b, 1.0, 1.0)
@benchmark mul!($c, $A_dense, $b, 1.0, 1.0)
