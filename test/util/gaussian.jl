using TemporalGPs: Gaussian

# This is a ridiculous definition that makes no sense. Don't use this anywhere.
Base.zero(x::Gaussian) = Gaussian(zero(x.m), zero(x.P))

function fd_isapprox(x_ad::Gaussian, x_fd::Gaussian, rtol, atol)
    return fd_isapprox(x_ad.m, x_fd.m, rtol, atol) &&
        fd_isapprox(x_ad.P, x_fd.P, rtol, atol)
end

@testset "gaussian" begin
    @testset "Gaussian" begin
        x = Gaussian(randn(3), randn(3, 3))
        x_vec, back = to_vec(x)
        @test back(x_vec) == x
    end
end
