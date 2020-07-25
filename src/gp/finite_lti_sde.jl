const FiniteLTISDE = FiniteGP{<:LTISDE}

#
# This is all a bit ugly, and would ideally go. IIRC there's some issue with the
# interactions between `FillArrays` and `Zygote` here that is problematic.
#

build_Σs(σ²_ns::AbstractVector{<:Real}) = SMatrix{1, 1}.(σ²_ns)

@adjoint function build_Σs(σ²_ns::Vector{<:Real})
    function build_Σs_Vector_back(Δ)
        return (first.(Δ),)
    end
    return build_Σs(σ²_ns), build_Σs_Vector_back
end

@adjoint function build_Σs(σ²_ns::Fill{<:Real})
    function build_Σs_Fill_back(Δ::NamedTuple)
        return ((value=first(Δ.value),),)
    end
    return build_Σs(σ²_ns), build_Σs_Fill_back
end

# Implement Stheno's version of the FiniteGP API. This will eventually become AbstractGPs
# API, but Stheno is still on a slightly different API because I've yet to update it.

function build_lgssm(ft::FiniteLTISDE)
    model = LGSSM(GaussMarkovModel(ft.f.f.k, ft.x, ft.f.storage), build_Σs(diag(ft.Σy)))
    return ScalarLGSSM(model)
end

Stheno.mean(ft::FiniteLTISDE) = mean(build_lgssm(ft))

Stheno.cov(ft::FiniteLTISDE) = cov(build_lgssm(ft))

Stheno.marginals(ft::FiniteLTISDE) = error("Not implemented despite it being easy.")

# We currently only implement one method of this 
Stheno.rand(rng::AbstractRNG, ft::FiniteLTISDE) = rand(rng, build_lgssm(ft))
Stheno.rand(ft::FiniteLTISDE) = rand(Random.GLOBAL_RNG, ft)

# Multi-argument version not implemented yet.
Stheno.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Real}) = logpdf(build_lgssm(ft), y)
