#
# Initial set up. This stuff should mean that this script will run on a fresh install.
#

using Pkg
Pkg.activate(".");
Pkg.instantiate();



#
# Load the various packages on which this script depends.
#

using Revise
using DrWatson, Stheno, BenchmarkTools, PGFPlotsX, ProgressMeter, TemporalGPs, Random,
    DataFrames, Zygote

using DrWatson: @dict, @tagsave

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

const exp_dir_name = "single_output_gps"
const data_dir = joinpath(datadir(), exp_dir_name)



build_gp(k_base, σ², l) = GP(σ² * stretch(k_base, 1 / l), GPC())

# Naive implementation.
function build(::Val{:naive}, k_base, σ², l, x, σ²_n)
    f = build_gp(k_base, σ², l)
    return f(x, σ²_n)
end

# Basic lgssm implementation.
function build(::Val{:heap}, k_base, σ², l, x, σ²_n)
    f = build_gp(k_base, σ², l)
    return to_sde(f)(x, σ²_n)
end

# Stack-allocated arrays lgssm implementation.
function build(::Val{:stack}, k_base, σ², l, x, σ²_n)
    f = build_gp(k_base, σ², l)
    return to_sde(f, SArrayStorage(Float64))(x, σ²_n)
end

# Basic lgssm implementation.
function build(::Val{:heapF32}, k_base, σ², l, x, σ²_n)
    f = build_gp(k_base, σ², l)
    return to_sde(f, ArrayStorage(Float32))(x, Float32(σ²_n))
end

# Stack-allocated arrays lgssm implementation.
function build(::Val{:stackF32}, k_base, σ², l, x, σ²_n)
    f = build_gp(k_base, σ², l)
    return to_sde(f, SArrayStorage(Float32))(x, Float32(σ²_n))
end

# Generic logpdf computation.
function build_and_logpdf(val::Val, k_base, σ², l, x, σ²_n, y)
    return logpdf(build(val, k_base, σ², l, x, σ²_n), y)
end



# Specify which experiments to run.
tagsave(
    joinpath(datadir(), exp_dir_name, "meta", "settings.bson"),
    Dict(
        # :Ns => [
        #     2, 5, 10,
        #     20, 50, 100,
        #     200, 500, 1_000,
        #     2_000, 5_000, 10_000,
        #     20_000, 50_000, 100_000,
        #     200_000, 500_000, 1_000_000,
        #     2_000_000, 5_000_000, 10_000_000,
        # ],
        :Ns => [
            2, 5, 10,
            # 20, 50, 100,
        ],
        :kernels => [
            # (k=Matern12(), sym=:Matern12, name="Matern12"),
            # (k=Matern32(), sym=:Matern32, name="Matern32"),
            (k=Matern52(), sym=:Matern52, name="Matern52"),
        ],
        :implementations => [
            (
                name="naive",
                val=Val(:naive),
                colour="blue",
                marker="square*",
            ),
            (
                name="heap",
                val=Val(:heap),
                colour="black",
                marker="*",
            ),
            (
                name="stack",
                val=Val(:stack),
                colour="red",
                marker="triangle*",
            ),
            (
                name="heap-F32",
                val=Val(:heapF32),
                colour="green",
                marker="*",
            ),
            (
                name="stack-F32",
                val=Val(:stackF32),
                colour="purple",
                marker="triangle*",
            ),
        ]
    )
)



# Run to execute all benchmarking experiments specified in settings.bson.
let
    settings = load(joinpath(datadir(), exp_dir_name, "meta", "settings.bson"))
    settings_list = dict_list(settings)
    progress = Progress(length(settings_list))
    for (p, setting) in enumerate(settings_list)

        N = setting[:Ns]
        kernel = setting[:kernels]
        impl = setting[:implementations]

        if N > 10_000 && impl.name == "naive"
            println("skipping naive implementation with N=$N")
            continue
        end

        # Construct data for problem.
        x = range(-5.0; length=N, step=1e-2)
        σ², l, σ²_n = 1.0, 2.3, 0.5
        k = kernel.k
        rng = MersenneTwister(123456)
        # y = rand(rng, build(Val(:stack), k, σ², l, x, σ²_n))
        y = rand(rng, build(impl.val, k, σ², l, x, σ²_n))

        # Generate results including construction of GP.
        results = @benchmark(build_and_logpdf($(impl.val), $k, $σ², $l, $x, $σ²_n, $y))
        grad_results = @benchmark(Zygote.gradient(
            $build_and_logpdf, $(impl.val), $k, $σ², $l, $x, $σ²_n, $y,
        ))

        # Generate results excluding construction of GP.
        fx = build(impl.val, k, σ², l, x, σ²_n)
        no_build_results = @benchmark(logpdf($fx, $y))
        no_build_grad_results = @benchmark(Zygote.gradient(logpdf, $fx, $y))

        # Save results in predictable location.
        @tagsave(
            joinpath(
                datadir(),
                exp_dir_name,
                savename(Dict(:N=>N, :k=>kernel.sym, :impl=>impl.name), "bson"),
            ),
            Dict(
                "setting" => setting,
                "results" => results,
                "grad_results" => grad_results,
                "no_build_results" => no_build_results,
                "no_build_grad_results" => no_build_grad_results,
                "N" => N,
            ),
        )

        next!(progress)
    end
end



# Run to generate figures specified in settings.bson.
let

    # Load + extract experimental settings.
    settings = load(joinpath(data_dir, "meta", "settings.bson"))
    colour_map = Dict(impl.name => impl.colour for impl in settings[:implementations])

    # Load and post-process results.
    results = DataFrame(
        map(eachrow(collect_results(data_dir))) do row
            row = merge(
                copy(row),
                (
                    kernel = row.setting[:kernels].name,
                    impl = row.setting[:implementations].name,
                    total_time = time(row.results) / 1e9,
                    grad_total_time = time(row.grad_results) / 1e9,
                    run_time = time(row.no_build_results) / 1e9,
                    grad_run_time = time(row.no_build_grad_results) / 1e9,
                ),
            )
            row = merge(
                row,
                (
                    build_time = row.total_time - row.run_time,
                    grad_build_time = row.grad_total_time - row.grad_run_time,
                    total_rate = row.total_time / row.N,
                    grad_total_rate = row.grad_total_time / row.N,
                    run_rate = row.run_time / row.N,
                    grad_run_rate = row.grad_run_time / row.N,
                )
            )
            row = merge(
                row,
                (
                    grad_ratio_total = row.grad_total_time / row.total_time,
                    grad_ratio_build = row.grad_build_time / row.build_time,
                    grad_ratio_run = row.grad_run_time / row.run_time,
                    build_fraction = row.build_time / row.total_time,
                    grad_build_fraction = row.grad_build_time / row.grad_total_time,
                ),
            )
            return row
        end
    )

    # List the things to generate plots for.
    plot_keys = [:total_time, :grad_total_time, :run_time, :grad_run_time, :build_time,
        :grad_build_time, :build_fraction, :grad_build_fraction, :grad_ratio_total,
        :grad_ratio_run, :grad_ratio_build, :total_rate, :grad_total_rate, :run_rate,
        :grad_run_rate,
    ]

    plots = by(results, [:kernel, :impl]) do result
        result = sort(result, :N)
        impl, kernel = first(result).impl, first(result).kernel
        return (
            label = "\\label {$(impl)-$(kernel)}",
            legend_entry = LegendEntry(kernel * " -- " * impl),
            Pair.(
                plot_keys,
                map(plot_keys) do key
                    @pgf PlotInc(
                        {
                            color = colour_map[impl],
                            mark_options={fill=colour_map[impl]}
                        },
                        Coordinates(result[!, :N], result[!, key]),
                    )
                end,
            )...,
        )
    end



    #
    # Plot stuff using the resources generated above.
    #

    # Place to store things to plot.
    plots_to_save = Vector{NamedTuple{(:name, :data)}}(undef, 0)

    x_ticks = 10.0 .^ (0:2:8)
    horz_sep = "20pt"

    # Generate total time / grad total time 1 x 2 group plot.
    for (field, grad_field, yticks) in (
        (:total_time, :grad_total_time, 10.0 .^ (-8:2:2)),
        (:run_time, :grad_run_time, 10.0 .^ (-8:2:2)),
        (:build_time, :grad_build_time, 10.0 .^ (-8:2:2)),
        (:total_rate, :grad_total_rate, 10.0 .^ (-8:2:-1)),
        (:run_rate, :grad_run_rate, 10.0 .^ (-8:2:-1)),
    )
        for (xmode, ymode, name_suffix) in (
            ("normal", "log", ""),
            ("log", "log", "_log"),
        )
            push!(
                plots_to_save,
                (
                    name=string(field) * name_suffix,
                    data=@pgf GroupPlot( # Generate group plot
                        {
                            group_style =
                            {
                                group_size="2 by 1",
                                horizontal_sep=horz_sep,
                                yticklabels_at="edge left",
                            },
                        },
                        { # Style for lhs plot
                            xmode=xmode,
                            ymode=ymode,
                            xlabel="N",
                            ylabel="time (s)",
                            # legend_to_name="legend:$(string(field))",
                            xmajorgrids,
                            ymajorgrids,
                            width="0.48\\textwidth",
                            ytick=yticks,
                            ymin=minimum(yticks),
                            ymax=maximum(yticks),
                            xtick=x_ticks,
                            xmin=0,
                            xmax=1e8,
                        },
                        plots[!, field], # lhs plot
                        { # style for rhs plot
                            xmode=xmode,
                            ymode=ymode,
                            xlabel="N",
                            xmajorgrids,
                            ymajorgrids,
                            width="0.48\\textwidth",
                            ytick=yticks,
                            ymin=minimum(yticks),
                            ymax=maximum(yticks),
                            xtick=x_ticks,
                            xmin=0,
                            xmax=1e8,
                            legend_style =
                            {
                                font = "\\tiny",
                                at = Coordinate(1 + 0.1, 1),
                                anchor = "north west",
                                legend_columns = 1,
                            },
                            transpose_legend,
                        },
                        plots[!, grad_field], # rhs plot
                        plots[!, :legend_entry],
                    )
                ),
            )
        end
    end

    # Plot proportion of time spent setting up in both forwards and reverse passes.
    let
        yticks = [0.0, 0.5, 1.0]
        push!(
            plots_to_save,
            (
                name = "build_fraction",
                data = @pgf GroupPlot( # Generate group plot
                    {
                        group_style =
                        {
                            group_size="2 by 1",
                            horizontal_sep=horz_sep,
                            xticklabels_at="edge bottom",
                            yticklabels_at="edge left",
                        },
                    },
                    { # Style for lhs plot
                        xmode="log",
                        ymode="normal",
                        xlabel="N",
                        legend_to_name="legend:build-fraction",
                        legend_columns=3,
                        transpose_legend,
                        xmajorgrids,
                        ymajorgrids,
                        width="0.48\\textwidth",
                        xtick=x_ticks,
                        ytick=yticks,
                        ymin=minimum(yticks),
                        ymax=maximum(yticks),
                    },
                    plots[!, :build_fraction], # lhs plot
                    plots[!, :legend_entry],
                    { # style for rhs plot
                        xmode="log",
                        ymode="normal",
                        xlabel="N",
                        xmajorgrids,
                        ymajorgrids,
                        width="0.48\\textwidth",
                        xtick=x_ticks,
                        ytick=yticks,
                        ymin=minimum(yticks),
                        ymax=maximum(yticks),
                    },
                    plots[!, :grad_build_fraction], # rhs plot
                )
            ),
        )
    end

    # Plot various ratios of logpdf to logpdf + gradient of logpdf evaluations.
    for (field, yticks) in [
        (:grad_ratio_total, 0:5:50),
        (:grad_ratio_run, vcat(0:5:50, 2)),
        (:grad_ratio_build, 0:5:50),
    ]
        push!(
            plots_to_save,
            (
                name=string(field),
                data = @pgf Axis(
                    {
                        xmode="log",
                        ymode="normal",
                        xlabel="N",
                        width="0.45\\textwidth",
                        xmajorgrids,
                        ymajorgrids,
                        xtick=x_ticks,
                        ytick=yticks,
                    },
                    plots[!, field],
                    # legend_entries,
                )
            )
        )
    end

    # Write results plots to results directory.
    plots_dir = joinpath(plotsdir(), exp_dir_name)
    mkpath(plots_dir)
    for plot in plots_to_save
        pgfsave(
            joinpath(plots_dir, plot.name * ".tex"),
            plot.data;
            include_preamble=true,
        )
    end
end
