using TemporalGPs: build_Σs, smooth, posterior_rand, DenseStorage, StaticStorage

_logistic(x) = 1 / (1 + exp(-x))

println("lti_sde:")
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
            (name="heteroscedastic noise", val=(_logistic.(randn(rng, N)) .+ 1e-1,)),
            (name="none", val=(),),
            (name="Diagonal", val=(Diagonal(_logistic.(randn(rng, N)) .+ 1e-1),)),
        )

        @testset "t=$(t.name), storage=$(storage.name), σ²=$(σ².name)" for
            t in ts,
            storage in storages,
            σ² in σ²s

            f = GP(k, GPC())
            ft = f(t.val, σ².val...)

            f_sde = to_sde(f, storage.val)
            ft_sde = f_sde(t.val, σ².val...)

            validate_dims(ft_sde)

            # These things are slow. Only helpful for verifying correctness.
            @test mean(ft) ≈ mean(ft_sde)
            @test cov(ft) ≈ cov(ft_sde)

            y = rand(MersenneTwister(123456), ft)
            y_sde = rand(MersenneTwister(123456), ft_sde)
            @test y ≈ y_sde

            @test logpdf(ft, y) ≈ logpdf(ft_sde, y)


            _, y_smooth, _ = smooth(ft_sde, y)

            # Check posterior marginals
            m_ssm = [first(y.m) for y in y_smooth]
            σ²_ssm = [first(y.P) for y in y_smooth]

            f′ = f | (ft ← y)
            f′_marginals = marginals(f′(t.val, 1e-12))
            m_exact = mean.(f′_marginals)
            σ²_exact = std.(f′_marginals).^2

            @test isapprox(m_ssm, m_exact; atol=1e-9, rtol=1e-9)
            @test isapprox(σ²_ssm, σ²_exact; atol=1e-9, rtol=1e-9)
        end
    end

    @testset "StaticArrays performance integration" begin
        rng = MersenneTwister(123456)
        f = to_sde(GP(Matern32(), GPC()), StaticStorage())
        σ²_n = 0.54

        t = range(0.1; step=0.11, length=1_000_000)
        ft = f(t, σ²_n)
        y = collect(rand(rng, ft))

        @testset "logpdf performance" begin
            Δlml = randn(rng)

            # Ensure that allocs is roughly independent of length(t).
            primal, fwd, rvs = benchmark_adjoint(logpdf, Δlml, ft, y; disp=false)
            @test allocs(primal) < 100
            @test allocs(fwd) < 100
            @test allocs(rvs) < 100
        end
        @testset "rand" begin
            Δy = randn(rng, length(t))

            # Ensure that allocs is roughly independent of length(t).
            primal, fwd, rvs = benchmark_adjoint(
                ft->rand(MersenneTwister(123456), ft), Δy, ft;
                disp=false,
            )
            @test allocs(primal) < 100
            @test allocs(fwd) < 100
            @test allocs(rvs) < 100
        end
    end
end
