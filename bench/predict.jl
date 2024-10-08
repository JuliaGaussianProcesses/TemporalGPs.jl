using Pkg
Pkg.activate(".")
Pkg.instantiate()

using BenchmarkTools, BlockDiagonals, DataFrames, DrWatson, FillArrays, Kronecker,
    LinearAlgebra, PGFPlotsX, Random, TemporalGPs

using TemporalGPs: predict, predict_pullback, AV, AM

const n_blas_threads = Sys.CPU_THREADS;
LinearAlgebra.BLAS.set_num_threads(n_blas_threads);

const exp_dir_name = "predict"



#
# Saving / loading utilities.
#

function load_settings()
    return load(joinpath(datadir(), exp_dir_name, "meta", "settings.bson"))[:settings]
end

load_results() = collect_results(joinpath(datadir(), exp_dir_name))



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

function dense_dynamics_constructor(rng, dim_lat, n_obs, n_blocks)
    D = dim_lat * n_blocks * n_obs
    A = randn(rng, D, D)
    a = randn(rng, D)
    Q = collect(Symmetric(randn(rng, D, D)))

    mf = randn(rng, D)
    Pf = Symmetric(randn(rng, D, D))

    Δmp = randn(rng, size(a))
    ΔPp = randn(rng, size(Pf))

    return Δmp, ΔPp, mf, Pf, A, a, Q
end

function block_diagonal_dynamics_constructor(rng, dim_lat, n_obs, n_blocks)
    total_dim_lat = dim_lat * n_obs
    A = BlockDiagonal([randn(rng, total_dim_lat, total_dim_lat) for _ in 1:n_blocks])
    a = randn(rng, size(A, 1))
    Q = BlockDiagonal([randn(rng, total_dim_lat, total_dim_lat) for _ in 1:n_blocks])

    mf = randn(rng, size(a))
    Pf = Symmetric(randn(rng, size(A)))

    Δmp = randn(rng, size(a))
    ΔPp = randn(rng, size(Pf))

    return Δmp, ΔPp, mf, Pf, A, a, Q
end

function block_diagonal_kronecker_dynamics_constructor(rng, dim_lat, n_obs, n_blocks)
    As = map(_ -> Eye{Float64}(n_obs) ⊗ randn(rng, dim_lat, dim_lat), 1:n_blocks)
    A = BlockDiagonal(As)
    a = randn(rng, size(A, 1))
    Q = BlockDiagonal([randn(rng, dim_lat * n_obs, dim_lat * n_obs) for _ in 1:n_blocks])

    mf = randn(rng, size(a))
    Pf = Symmetric(randn(rng, size(A)))

    Δmp = randn(rng, size(a))
    ΔPp = randn(rng, size(Pf))

    return Δmp, ΔPp, mf, Pf, A, a, Q
end



#
# Save settings for benchmarking experiments.
#

wsave(
    joinpath(datadir(), exp_dir_name, "meta", "settings.bson"),
    :settings => dict_list(
        Dict(

            # Method to execute `predict` and construct / evaluate its pullback.
            :implementation =>[
                (
                    name = "naive",
                    predict = naive_predict,
                    predict_pullback = naive_predict_pullback,
                    dynamics_constructor = dense_dynamics_constructor,
                ),
                (
                    name = "dense",
                    predict = predict,
                    predict_pullback = predict_pullback,
                    dynamics_constructor = dense_dynamics_constructor,
                ),
                (
                    name = "block-diagonal",
                    predict = predict,
                    predict_pullback = predict_pullback,
                    dynamics_constructor = block_diagonal_dynamics_constructor,
                ),
                (
                    name = "block-diagonal-kronecker",
                    predict = predict,
                    predict_pullback = predict_pullback,
                    dynamics_constructor = block_diagonal_kronecker_dynamics_constructor,
                ),
            ],

            # Latent dimensionality for each observation.
            # :dim_lat => [5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            :dim_lat => [3],

            # Number of observations per time step.
            # :n_obs => [2, 5, 10, 25, 50, 100],
            :n_obs => [200, 300, 400, 500],

            # Number of blocks.
            # :n_blocks => [1, 2, 3, 4, 5],
            :n_blocks => [2, 3, 4],
        ),
    ),
)



#
# Run benchmarking experiments.
#

let
    settings = load_settings()
    N = length(settings)
    for (n, setting) in enumerate(settings)

        # Display current iteration.
        impl = setting[:implementation]
        dim_lat = setting[:dim_lat]
        n_obs = setting[:n_obs]
        n_blocks = setting[:n_blocks]
        println(
            "$n / $N: name=$(impl.name), dim_lat=$(dim_lat), "*
            "n_obs=$(n_obs), n_blocks=$(n_blocks)",
        )

        # Build dynamics model.
        rng = MersenneTwister(123456)
        Δmp, ΔPp, mf, Pf, A, a, Q = impl.dynamics_constructor(rng, dim_lat, n_obs, n_blocks)

        # Generate pullback.
        _, back = impl.predict_pullback(mf, Pf, A, a, Q)

        # Benchmark evaluation, pullback generation, and pullback evaluation.
        predict_results = @benchmark $(impl.predict)($mf, $Pf, $A, $a, $Q)
        generate_pullback_results =
            (@benchmark $(impl.predict_pullback)($mf, $Pf, $A, $a, $Q))
        pullback_results = @benchmark $back(($Δmp, $ΔPp))

        # Save results to disk.
        wsave(
            joinpath(
                datadir(),
                exp_dir_name,
                savename(
                    Dict(
                        :name => impl.name,
                        :dim_lat => dim_lat,
                        :n_obs => n_obs,
                        :n_blocks => n_blocks,
                    ),
                    "bson",
                ),
            ),
            Dict(
                :setting => setting,
                :predict_results => predict_results,
                :generate_pullback_results => generate_pullback_results,
                :pullback_results => pullback_results,
                :n_blas_threads => n_blas_threads,
                :n_julia_threads => Threads.nthreads(),
            ),
        )
    end
end



#
# Process results.
#

let
    settings = load_settings()

    # Define some plotting settings for each of the implementation types.
    colour_map = Dict(
        "naive" => "black",
        "dense" => "blue",
        "block-diagonal" => "red",
        "block-diagonal-kronecker" => "green",
    )
    marker_map = Dict(
        "naive" => "square",
        "dense" => "triangle",
        "block-diagonal" => "*",
        "block-diagonal-kronecker" => "circle",
    )

    try
        mkdir(plotsdir())
    catch
    end

    try
        mkdir(joinpath(plotsdir(), exp_dir_name))
    catch
    end

    # Load the results and compute some additional columns.
    results = DataFrame(
        map(eachrow(load_results())) do row
            row = merge(
                copy(row), # No need to copy in following blocks.
                (
                    implementation_name = row.setting[:implementation].name,
                    dim_lat = row.setting[:dim_lat] * row.setting[:n_obs],
                    n_blocks = row.setting[:n_blocks],
                ),
            )
            return row
        end
    )

    # Generate plots of timings per n_blocks per implementation.
    by(results, :n_blocks) do n_blocks_group

        # Specify plots to produce.
        plots = [
            (
                plot = (@pgf Axis(
                    {
                        title = "predict",
                        legend_pos = "north west",
                        xmode="log",
                        ymode="log",
                    }
                )),
                result_name = :predict_results,
                save_name = "predict_n_blocks=$(first(n_blocks_group.n_blocks)).tex",
            ),
            (
                plot = (@pgf Axis(
                    {
                        title = "forwards pass",
                        legend_pos = "north west",
                        xmode="log",
                        ymode="log",
                    },
                )),
                result_name = :generate_pullback_results,
                save_name = "generate_n_blocks=$(first(n_blocks_group.n_blocks)).tex",
            ),
            (
                plot = (@pgf Axis(
                    {
                        title = "reverse pass",
                        legend_pos = "north west",
                        xmode="log",
                        ymode="log",
                    },
                )),
                result_name = :pullback_results,
                save_name = "pullback_n_blocks=$(first(n_blocks_group.n_blocks)).tex",
            ),
        ]

        # Compute dim_lat vs time curves for each implementation, for each benchmarked op.
        by(n_blocks_group, :implementation_name) do impl_group

            impl_group = sort(impl_group, :dim_lat)

            impl = first(impl_group.implementation_name)

            for plt in plots
                push!(
                    plt.plot,
                    (@pgf Plot(
                        {
                            color = colour_map[impl],
                            mark_options={fill=colour_map[impl]},
                            mark=marker_map[impl],
                        },
                        Coordinates(
                            impl_group.dim_lat,
                            time.(impl_group[!, plt.result_name]),
                        ),
                    )),
                )
                push!(plt.plot, LegendEntry(impl))
            end
        end

        # Save plots.
        for plt in plots
            pgfsave(
                joinpath(plotsdir(), exp_dir_name, plt.save_name),
                plt.plot;
                include_preamble=true,
            )
        end
    end

    # Generate plots of timings per n_blocks per implementation.
    by(results, :dim_lat) do dim_lat_group

        # Specify plots to produce.
        plots = [
            (
                plot = (@pgf Axis(
                    {
                        title = "predict",
                        legend_pos = "north west",
                    }
                )),
                result_name = :predict_results,
                save_name = "predict_dim_lat=$(first(dim_lat_group.dim_lat)).tex",
            ),
            (
                plot = (@pgf Axis(
                    {
                        title = "forwards pass",
                        legend_pos = "north west",
                    },
                )),
                result_name = :generate_pullback_results,
                save_name = "generate_pb_dim_lat=$(first(dim_lat_group.dim_lat)).tex",
            ),
            (
                plot = (@pgf Axis(
                    {
                        title = "reverse pass",
                        legend_pos = "north west",
                    },
                )),
                result_name = :pullback_results,
                save_name = "pullback_dim_lat=$(first(dim_lat_group.dim_lat)).tex",
            ),
        ]

        # Compute dim_lat vs time curves for each implementation, for each benchmarked op.
        by(dim_lat_group, :implementation_name) do impl_group

            impl_group = sort(impl_group, :n_blocks)

            impl = first(impl_group.implementation_name)

            for plt in plots
                push!(
                    plt.plot,
                    (@pgf Plot(
                        {
                            color = colour_map[impl],
                            mark_options={fill=colour_map[impl]},
                            mark=marker_map[impl],
                        },
                        Coordinates(
                            impl_group.n_blocks,
                            time.(impl_group[!, plt.result_name]),
                        ),
                    )),
                )
                push!(plt.plot, LegendEntry(impl))
            end
        end

        # Save plots.
        for plt in plots
            pgfsave(
                joinpath(plotsdir(), exp_dir_name, plt.save_name),
                plt.plot;
                include_preamble=true,
            )
        end
    end
end






#
# Rough old benchmarks and profiling. Helpful for debugging performance, less so for
# producing nice graphs.
#


#
# Roughly benchmark Kronecker against Dense.
#

using BenchmarkTools, FillArrays, Kronecker, LinearAlgebra, Random, Stheno,
    TemporalGPs, Mooncake

using TemporalGPs: predict

rng = MersenneTwister(123456);
D = 3;
N = 247;

# Compute the total number of dimensions.
Dlat = N * D;

# Generate Kronecker-Product transition dynamics.
A_D = randn(rng, Float64, D, D);
A = Eye{Float64}(N) ⊗ A_D;

a = randn(rng, Float64, Dlat);

Q = randn(rng, Dlat, Dlat);

# Generate filtering (input) distribution.
mf = randn(rng, Float64, Dlat);
Pf = Symmetric(Stheno.kernelmatrix(SEKernel(), range(-10.0, 10.0; length=Dlat)));


# Generate corresponding dense dynamics.
A_dense = collect(A);

@benchmark predict($mf, $Pf, $A, $a, $Q)
@benchmark predict($mf, $Pf, $A_dense, $a, $Q)

# using ProfileView
# @profview [predict(mf, Pf, A, a, Q) for _ in 1:10]

@benchmark Mooncake.pullback(predict, $mf, $Pf, $A, $a, $Q)
@benchmark Mooncake.pullback(predict, $mf, $Pf, $A_dense, $a, $Q)

_, back = Mooncake.pullback(predict, mf, Pf, A, a, Q);
_, back_dense = Mooncake.pullback(predict, mf, Pf, A_dense, a, Q);

mp = copy(mf);
Pp = collect(Pf);

@benchmark $back(($mp, $Pp))
@benchmark $back_dense(($mp, $Pp))

# using ProfileView
# @profview [back((mp, Pp)) for _ in 1:10]


T = Float64;
Δmp = copy(mp);
ΔPp = copy(Pp);
Δmf = fill(zero(T), size(mf));
ΔPf = fill(zero(T), size(Pf));
ΔA = TemporalGPs.get_cotangent_storage(A, zero(T));
Δa = fill(zero(T), size(a));
ΔQ = TemporalGPs.get_cotangent_storage(Q, zero(T));
@benchmark TemporalGPs.predict_pullback_accum!(
    $Δmp, $ΔPp, $Δmf, $ΔPf, $ΔA, $Δa, $ΔQ,
    $mf, $Pf, $A, $a, $Q,
)

# @profview [TemporalGPs.predict_pullback_accum!(
#     Δmp, ΔPp, Δmf, ΔPf, ΔA, Δa, ΔQ,
#     mf, Pf, A, a, Q,
# ) for _ in 1:100]



#
# Roughly benchmark BlockDiagonal of Kroneckers against Dense.
#

using BenchmarkTools, BlockDiagonals, FillArrays, Kronecker, LinearAlgebra, Random, Stheno,
    TemporalGPs, Mooncake

using TemporalGPs: predict



rng = MersenneTwister(123456);
D = 3;
N = 247;
N_blocks = 3;
T = Float64;


Dlat = N * D * N_blocks;

# Generate BlockDiagonal-KroneckerProduct transition dynamics.
A_Ds = [randn(rng, T, D, D) for _ in 1:N_blocks];
As = [Eye{T}(N) ⊗ A_Ds[n] for n in 1:N_blocks];
A = BlockDiagonal(As);

a = randn(rng, T, N * D * N_blocks);

Qs = [randn(rng, T, N * D, N * D) for _ in 1:N_blocks];
Q = BlockDiagonal(Qs);

# Generate filtering (input) distribution.
mf = randn(rng, T, Dlat);
Pf_ = randn(rng, T, Dlat, Dlat);
Pf = Symmetric(Pf_'Pf_ + I);


# Generate corresponding dense dynamics.
A_dense = collect(A);
Q_dense = collect(Q);

@benchmark predict($mf, $Pf, $A, $a, $Q)
@benchmark predict($mf, $Pf, $A_dense, $a, $Q_dense)

# using ProfileView
# @profview [predict(mf, Pf, A, a, Q) for _ in 1:10]

@benchmark Mooncake.pullback(predict, $mf, $Pf, $A, $a, $Q)
@benchmark Mooncake.pullback(predict, $mf, $Pf, $A_dense, $a, $Q_dense)

_, back = Mooncake.pullback(predict, mf, Pf, A, a, Q);
_, back_dense = Mooncake.pullback(predict, mf, Pf, A_dense, a, Q_dense);

mp = copy(mf);
Pp = collect(Pf);

@benchmark $back(($mp, $Pp))
@benchmark $back_dense(($mp, $Pp))
