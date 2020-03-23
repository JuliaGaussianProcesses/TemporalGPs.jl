using TemporalGPs: build_Σs

_logistic(x) = 1 / (1 + exp(-x))

@testset "lti_sde" begin
    @testset "build_Σs" begin
        rng = MersenneTwister(123456)
        N = 11
        @testset "heteroscedastic" begin
            σ²_ns = exp.(randn(rng, N)) .+ 1e-3
            @test all(first.(build_Σs(σ²_ns)) == first.(σ²_ns))

            ΔΣs = SMatrix{1, 1}.(randn(rng, N))
            adjoint_test(build_Σs, ΔΣs, σ²_ns)
        end
        @testset "homoscedastic" begin
            σ²_n = exp(randn(rng)) + 1e-3
            σ²_ns = Fill(σ²_n, N)
            @test all(first.(build_Σs(σ²_ns)) == first.(σ²_ns))

            ΔΣs = (value=SMatrix{1, 1}(randn(rng)),)
            adjoint_test(σ²_n->build_Σs(Fill(σ²_n, N)), ΔΣs, σ²_n)
        end
    end

    @testset "GP correctness" begin

        rng = MersenneTwister(123456)
        k = Matern32()
        N = 7

        # construct a Gauss-Markov model with either dense storage or static storage.
        storages = (
            (name="dense storage", val=DenseStorage()),
            (name="static storage", val=StaticStorage()),
        )

        # Either regular spacing or irregular spacing in time.
        ts = (
            (name="irregular spacing", val=sort(rand(rng, N))),
            # (name="regular spacing", val=range(0.0; step=0.3, length=N)),
        )

        σ²s = (
            (name="homoscedastic noise", val=(0.1,)),
            (name="heteroscedastic noise", val=(_logistic.(-randn(rng, N)) .+ 1e-1,)),
            (name="none", val=(),),
            (name="Diagonal", val=(Diagonal(_logistic.(-randn(rng, N)) .+ 1e-1),)),
        )

        @testset "t=$(t.name), storage=$(storage.name), σ²=$(σ².name)" for
            t in ts,
            storage in storages,
            σ² in σ²s

            f = GP(k, GPC())
            ft = f(t.val, σ².val...)

            f_sde = to_sde(f, storage.val)
            ft_sde = f_sde(t.val, σ².val...)

            # These things are slow. Only helpful for verifying correctness.
            @test mean(ft) ≈ mean(ft_sde)
            @test cov(ft) ≈ cov(ft_sde)

            y = rand(MersenneTwister(123456), ft)
            y_sde = rand(MersenneTwister(123456), ft_sde)
            @test y ≈ y_sde

            @test logpdf(ft, y) ≈ logpdf(ft_sde, y)
        end

    end

    # _, y_smooth, _ = smooth(ssm(f(t.val, σ².val), storage.val), y)

    # # Check posterior marginals
    # m_ssm = [first(y.m) for y in y_smooth]
    # σ²_ssm = [first(y.P) for y in y_smooth]

    # f′ = f | (f(t.val, σ².val) ← y)
    # f′_marginals = marginals(f′(t.val))
    # m_exact = mean.(f′_marginals)
    # σ²_exact = std.(f′_marginals).^2

    # @test m_ssm ≈ m_exact
    # @test σ²_ssm ≈ σ²_exact
end
