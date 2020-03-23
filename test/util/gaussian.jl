using TemporalGPs: Gaussian

# This is a ridiculous definition that makes no sense. Don't use this anywhere.
Base.zero(x::Gaussian) = Gaussian(zero(x.m), zero(x.P))

function fd_isapprox(x_ad::Gaussian, x_fd::Gaussian, rtol, atol)
    return fd_isapprox(x_ad.m, x_fd.m, rtol, atol) &&
        fd_isapprox(x_ad.P, x_fd.P, rtol, atol)
end

# @testset "Gaussian" begin
#     rng = MersenneTwister(123456)
#     N = 11

#     let
#         x = Gaussian(randn(rng, N), randn(rng, N, N))
#         y = Gaussian(randn(rng, N), randn(rng, N, N))
#     end
# end
