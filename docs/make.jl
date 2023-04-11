using TemporalGPs
using Pkg
Pkg.add(Pkg.PackageSpec(; url="https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl")) # While the package is unregistered, it's a workaround

using JuliaGPsDocs

JuliaGPsDocs.generate_examples(TemporalGPs; ntasks=3)

using Documenter

makedocs(;
    modules=[TemporalGPs],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Examples" => JuliaGPsDocs.find_generated_examples(KernelFunctions),
    ],
    repo="https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/blob/{commit}{path}#L{line}",
    sitename="TemporalGPs.jl",
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    strict=true,
    checkdocs=:exports,
    doctestfilters=JuliaGPsDocs.DOCTEST_FILTERS,
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/TemporalGPs.jl", push_preview=true
)
