using TemporalGPs: build_Σs, smooth, posterior_rand

_logistic(x) = 1 / (1 + exp(-x))

# Some hacks to ensure element types make sense.
_to_T(T) = ()
_to_T(T, x::Real) = (T(x),)
_to_T(T, x::AbstractVector{<:Real}) = (T.(x),)
_to_T(T, X::Diagonal{<:Real}) = (T.(X),)

println("lti_sde:")
@testset "lti_sde" begin
    @testset "build_Σs" begin
        rng = MersenneTwister(123456)
        N = 11
        @testset "heteroscedastic" begin
            σ²_ns = exp.(randn(rng, N)) .+ 1e-3
            @test all(first.(build_Σs(σ²_ns)) == first.(σ²_ns))

            ΔΣs = SMatrix{1, 1}.(randn(rng, N))
            adjoint_test(build_Σs, (σ²_ns, ))
        end
        @testset "homoscedastic" begin
            σ²_n = exp(randn(rng)) + 1e-3
            σ²_ns = Fill(σ²_n, N)
            @test all(first.(build_Σs(σ²_ns)) == first.(σ²_ns))

            ΔΣs = (value=SMatrix{1, 1}(randn(rng)),)
            adjoint_test(σ²_n->build_Σs(Fill(σ²_n, N)), (σ²_n, ))
        end
    end

    @testset "GP correctness" begin

        rng = MersenneTwister(123456)
        k = Matern32()
        N = 7

        # construct a Gauss-Markov model with either dense storage or static storage.
        storages = (
            (name="dense storage Float64", val=ArrayStorage(Float64), tol=1e-9),
            (name="static storage Float64", val=SArrayStorage(Float64), tol=1e-9),
            (name="dense storage Float32", val=ArrayStorage(Float32), tol=1e-4),
            (name="static storage Float32", val=SArrayStorage(Float32), tol=1e-4),
        )

        # Either regular spacing or irregular spacing in time.
        ts = (
            (name="irregular spacing", val=sort(rand(rng, N))),
            (name="regular spacing", val=RegularSpacing(0.0, 0.3, N)),
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

            s = _to_T(eltype(storage.val), σ².val...)

            f = GP(k, GPC())
            ft = f(t.val, s...)

            f_sde = to_sde(f, storage.val)
            ft_sde = f_sde(t.val, s...)
            lgssm = TemporalGPs.build_lgssm(ft_sde)

            hetero_noise = σ².val isa Tuple{Union{Vector, Diagonal}}
            should_be_time_invariant = (t.val isa Vector || hetero_noise) ? false : true
            @test is_time_invariant(lgssm) == should_be_time_invariant

            validate_dims(lgssm)

            # These things are slow. Only helpful for verifying correctness.
            @test mean(ft) ≈ mean(ft_sde)
            @test cov(ft) ≈ cov(ft_sde)

            y = rand(MersenneTwister(123456), ft)
            y_sde = rand(MersenneTwister(123456), ft_sde)
            @test y ≈ y_sde

            @test logpdf(ft, y) ≈ logpdf(ft_sde, y_sde)

            if eltype(storage.val) == Float64
                if t.val isa Vector
                    adjoint_test(
                        (t, y) -> begin
                            _f = to_sde(GP(k, GPC()))
                            _ft = f(t, s...)
                            return logpdf(_ft, y)
                        end,
                        (t.val, y);
                        check_infers=false,
                    )
                else
                    adjoint_test(
                        (Δt, y) -> begin
                            _t = RegularSpacing(t.val.t0, Δt, length(t.val))
                            _f = to_sde(GP(k, GPC()))
                            _ft = _f(_t, s...)
                            return logpdf(_ft, y)
                        end,
                        (t.val.Δt, y);
                        check_infers=false,
                    )
                end
            end

            _, y_smooth, _ = smooth(lgssm, y_sde)

            # Check posterior marginals
            m_ssm = [first(y.m) for y in y_smooth]
            σ²_ssm = [first(y.P) for y in y_smooth]

            f′ = f | (ft ← y)
            f′_marginals = marginals(f′(t.val, 1e-12))
            m_exact = mean.(f′_marginals)
            σ²_exact = std.(f′_marginals).^2

            tol = storage.tol
            @test isapprox(m_ssm, m_exact; atol=tol, rtol=tol)
            @test isapprox(σ²_ssm, σ²_exact; atol=tol, rtol=tol)
        end
    end

    @testset "StaticArrays performance integration" begin
        rng = MersenneTwister(123456)
        f = to_sde(GP(Matern32(), GPC()), SArrayStorage(Float64))
        σ²_n = 0.54

        t = range(0.1; step=0.11, length=1_000)
        ft = f(t, σ²_n)
        y = collect(rand(rng, ft))

        @testset "logpdf performance" begin
            Δlml = randn(rng)

            # Ensure that allocs is roughly independent of length(t).
            primal, fwd, rvs = benchmark_adjoint(logpdf, Δlml, ft, y; disp=false)
            @test allocs(primal) < 200
            @test allocs(fwd) < 200
            @test allocs(rvs) < 200
        end
        @testset "rand" begin
            Δy = randn(rng, length(t))

            # Ensure that allocs is roughly independent of length(t).
            primal, fwd, rvs = benchmark_adjoint(
                ft->rand(MersenneTwister(123456), ft), Δy, ft;
                disp=false,
            )
            @test allocs(primal) < 200
            @test allocs(fwd) < 200
            @test allocs(rvs) < 200
        end
    end
end
