using Pkg
Pkg.activate(".")
Pkg.instantiate()

using BenchmarkTools, BlockDiagonals, FillArrays, LinearAlgebra, Random, Stheno,
    TemporalGPs, Zygote

using DataFrames, DrWatson, PGFPlotsX

using TemporalGPs: AV, AM, Separable, RectilinearGrid, LGSSM, GaussMarkovModel

# const n_blas_threads = Sys.CPU_THREADS;
const n_blas_threads = 4;
BLAS.set_num_threads(n_blas_threads);

const exp_dir_name = "lgssm"



#
# Utilities
#

function load_settings()
    return load(joinpath(datadir(), exp_dir_name, "meta", "settings.bson"))[:settings]
end

load_results() = collect_results(joinpath(datadir(), exp_dir_name))

function dense_dynamics_constructor(rng, N_space, N_time, N_blocks)

    # Build kernel with block-diagonal transition dynamics.
    ft = block_diagonal_dynamics_constructor(rng, N_space, N_time, N_blocks)

    # Densify dynamics and construct new LGSSM.
    return LGSSM(
        GaussMarkovModel(
            Fill(collect(first(ft.gmm.A)), length(ft)),
            ft.gmm.a,
            Fill(collect(first(ft.gmm.Q)), length(ft)),
            ft.gmm.H,
            ft.gmm.h,
            ft.gmm.x0,
        ),
        ft.Σ,
    )
end

function block_diagonal_dynamics_constructor(rng, N_space, N_time, N_blocks)

    # Construct kernel.
    k = Separable(EQ(), Matern52())
    for n in 1:(N_blocks - 1)
        k += k
    end

    f = to_sde(GP(k, GPC()))
    t = range(-5.0, 5.0; length=N_time)
    x = randn(rng, N_space)
    return f(RectilinearGrid(x, t), 0.1)
end



#
# Save settings.
#

wsave(
    joinpath(datadir(), exp_dir_name, "meta", "settings.bson"),
    :settings => dict_list(
        Dict(

            # Method to execute `predict` and construct / evaluate its pullback.
            :implementation => [
                (
                    name = "naive",
                    dynamics_constructor = dense_dynamics_constructor,
                ),
                (
                    name = "dense",
                    dynamics_constructor = dense_dynamics_constructor,
                ),
                (
                    name = "block-diagonal",
                    dynamics_constructor = block_diagonal_dynamics_constructor,
                ),
            ],

            # Number of observations in space.
            :N_space => [247],

            # Number of time points.
            # :N_time => [2, 5, 10, 50, 100, 500, 1_000, 50_000],
            :N_time => [25, 50, 75, 100],

            # Number of things to sum in the kernel.
            :N_blocks => [1, 2, 3],
        ),
    ),
)



#
# Run benchmarking experiments.
#

let
    settings = load_settings()
    N_settings = length(settings)
    for (n, setting) in enumerate(settings)

        # Display current iteration.
        impl = setting[:implementation]
        N_space = setting[:N_space]
        N_time = setting[:N_time]
        N_blocks = setting[:N_blocks]
        println(
            "$n / $N_settings: name=$(impl.name), N_space=$N_space, " *
            "N_time=$N_time, N_blocks = $N_blocks",
        )

        # Build dynamics model.
        rng = MersenneTwister(123456)
        ft = impl.dynamics_constructor(rng, N_space, N_time, N_blocks)
        y = rand(rng, ft)

        # Benchmark rand evaluation.
        rand_results = @benchmark rand($rng, $ft)

        # Benchmark logpdf evaluation and gradient evaluation.
        logpdf_results = @benchmark logpdf($ft, $y)
        logpdf_gradient_results = @benchmark Zygote.gradient(logpdf, $ft, $y)

        # Save results to disk.
        wsave(
            joinpath(
                datadir(),
                exp_dir_name,
                savename(
                    Dict(
                        :name => impl.name,
                        :N_space => N_space,
                        :N_time => N_time,
                        :N_blocks => N_blocks,
                    ),
                    "bson",
                ),
            ),
            Dict(
                :setting => setting,
                :rand_results => rand_results,
                :logpdf_results => logpdf_results,
                :logpdf_gradient_results => logpdf_gradient_results,
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
    # Define some plotting settings for each of the implementation types.
    colour_map = Dict(
        "naive" => "black",
        "dense" => "blue",
        "block-diagonal" => "red",
    )
    marker_map = Dict(
        "naive" => "square",
        "dense" => "triangle",
        "block-diagonal" => "*",
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
                    N_space = row.setting[:N_space],
                    N_time = row.setting[:N_time],
                    N_blocks = row.setting[:N_blocks],
                ),
            )
            return row
        end
    )

    # Plot timing curves for each block size for each implementation. This is used to assess
    # after what value of N_time the time-per-iteration saturates.
    by(results, [:N_blocks, :implementation_name]) do group

        # Specify plots to produce.
        N_blocks = first(group.N_blocks)
        impl_name = first(group.implementation_name)
        setting_name = "N_blocks=$N_blocks-impl_name=$impl_name"
        plots = [
            (
                plot = (@pgf Axis(
                    {
                        title = "rand",
                        legend_pos = "north west",
                        xmode="log",
                    }
                )),
                result_name = :rand_results,
                save_name = "rand_$setting_name.tex",
            ),
            (
                plot = (@pgf Axis(
                    {
                        title = "logpdf",
                        legend_pos = "north west",
                        xmode="log",
                    },
                )),
                result_name = :logpdf_results,
                save_name = "logpdf_$setting_name.tex",
            ),
            (
                plot = (@pgf Axis(
                    {
                        title = "logpdf gradient",
                        legend_pos = "north west",
                        xmode="log",
                    },
                )),
                result_name = :logpdf_gradient_results,
                save_name = "logpdf_gradient_$setting_name.tex",
            ),
        ]

        by(group, :N_space) do space_group

            space_group = sort(space_group, :N_time)
            for plt in plots
                push!(
                    plt.plot,
                    (@pgf Plot(
                        {
                            color = colour_map[impl_name],
                            mark_options={fill=colour_map[impl_name]},
                            mark=marker_map[impl_name],
                        },
                        Coordinates(
                            space_group.N_time,
                            time.(space_group[!, plt.result_name]) ./ space_group.N_time,
                        ),
                    )),
                )
                push!(plt.plot, LegendEntry(string(first(space_group.N_space))))
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
# Hacked together benchmarks for playing around.
#

rng = MersenneTwister(123456);
Ts = [1, 10, 100, 1_000];
N_space = 500;
N_blocks = 1;

model = dense_dynamics_constructor(rng, N_space, 100, N_blocks);
y = rand(rng, model);
@benchmark rand($rng, $model)
@benchmark logpdf($model, $y)

using Profile, ProfileView

@profview logpdf(model, y)
@profview logpdf(model, y)



# Test simple things quickly.
rng = MersenneTwister(123456);
T = 1_000_000;
x = range(0.0; step=0.3, length=T);
f = GP(Matern52() + Matern52() + Matern52() + Matern52(), GPC());
fx_sde_dense = to_sde(f)(x);
fx_sde_static = to_sde(f, SArrayStorage(Float64))(x);

y = rand(fx_sde_static);
@benchmark logpdf($fx_sde_dense, $y)
@benchmark logpdf($fx_sde_static, $y)


using Profile, ProfileView

@profview logpdf(fx_sde_dense, y)
@profview logpdf(fx_sde_dense, y)
