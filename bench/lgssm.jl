using BenchmarkTools, BlockDiagonals, FillArrays, LinearAlgebra, Random, Stheno,
    TemporalGPs, Zygote

using TemporalGPs: predict, predict_pullback, AV, AM, Separable, RectilinearGrid, LGSSM,
    GaussMarkovModel

BLAS.set_num_threads(16);



#
# Generate an LGSSM with BlockDiagonal transition dynamics, comprising three blocks of
# equal size.
#

rng = MersenneTwister(123456)

k = Separable(EQ(), Matern52());

f = to_sde(GP(k + k + k, GPC()));
t = range(-5.0, 5.0; length=100);
x = randn(rng, 247);

ft = f(RectilinearGrid(x, t), 0.1);
y = rand(rng, ft);

println("Construction")
display(@benchmark $f(RectilinearGrid($x, $t), 0.1))
println()

println("rand")
display(@benchmark rand($rng, $ft))
println()

ft_dense = LGSSM(
    GaussMarkovModel(
        Fill(collect(first(ft.gmm.A)), length(t)),
        ft.gmm.a,
        Fill(collect(first(ft.gmm.Q)), length(t)),
        ft.gmm.H,
        ft.gmm.h,
        ft.gmm.x0,
    ),
    ft.Σ,
);

println("rand (dense)")
display(@benchmark rand($rng, $ft_dense))
println()

println("logpdf")
display(@benchmark logpdf($ft, $y))
println()

println("logpdf (dense)")
display(@benchmark logpdf($ft_dense, $y))
println()
