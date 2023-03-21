using Documenter, TemporalGPs

makedocs(;
    modules=[TemporalGPs],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/blob/{commit}{path}#L{line}",
    sitename="TemporalGPs.jl",
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/TemporalGPs.jl",
)
