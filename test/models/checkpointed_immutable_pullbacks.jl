using TemporalGPs:
    Immutable,
    Checkpointed,
    correlate,
    decorrelate,
    correlate_pullback,
    decorrelate_pullback,
    copy_first

@testset "checkpointed_immutable_pullbacks" begin

    # Construct an LGSSM.
    rng = MersenneTwister(123456)
    Dlat = 3
    Dobs = 2
    N = 10
    storage = ArrayStorage(Float64)
    model = random_ti_lgssm(rng, Dlat, Dobs, N, storage)

    # Generate data for correlate / decorrelate operations.
    y = rand(model)
    _, α = TemporalGPs.decorrelate(model, y)

    # Perform filtering / gradient propagation with no checkpointing.
    (lml_naive, ys_naive), pb_naive = correlate_pullback(Immutable(), model, α, copy_first)

    # Perform filtering / gradient propagation with checkpointing.
    (lml_checkpoint, ys_checkpoint), pb_checkpoint = correlate_pullback(
        Checkpointed(),
        Immutable(),
        model,
        α,
        copy_first,
    )

    @test lml_naive ≈ lml_checkpoint
    @test ys_naive == ys_checkpoint

    Δlml = randn()
    Δys = [randn(Dobs) for _ in 1:N]

    _, Δmodel_naive, Δαs_naive, _ = pb_naive((Δlml, Δys))
    _, _, Δmodel_checkpoint, Δαs_checkpoint = pb_checkpoint((Δlml, Δys))

    @test Δmodel_naive == Δmodel_checkpoint
    @test Δαs_naive == Δαs_checkpoint
end
