using AbstractGPs
using TemporalGPs
using Distributions
using StatsFuns
using LinearAlgebra
using ProgressMeter

# Utilising TemporalGPs.jl to work with AbstractGPs requires minimal modification to the
# GP objects from AbstractGPs that you are used to. The primary differences are
# 1. RegularSpacing is a useful type. It's basically a `range` that's hacked together to
#    work nicely with Zygote.jl. At some point, it will hopefully disappear.
# 2. Call `to_sde` on your AbstractGP object to say "use TemporalGPs.jl to do inference".

# This is an example of a very, very noise regression problem.
# You would probably be better of using a pseudo-point approximation for this particular
# data set, but longer time-series tend to lend themselves less well to nice plots.

# Load up the separable kernel from TemporalGPs.
using TemporalGPs: RegularSpacing

# Build a GP as per usual, and wrap it inside a TemporalGPs.jl object.
f_raw = GP(Matern52Kernel());
f = to_sde(f_raw, SArrayStorage(Float64));

# Specify a collection of inputs. Must be increasing.
T = 1_000;
x = RegularSpacing(0.0, 1e-1, T);

# Generate some noisy synthetic data from the GP.
σ²_noise = 0.01;
f_true = rand(f(x, σ²_noise))
y = rand.(Bernoulli.(logistic.(f_true)));
y_sign = sign.(y .- 0.5)

plt = plot()
scatter!(plt, x, y; label="", markersize=1.0, alpha=0.1)
plot!(plt, x, f_true; label="", lw=2.0)

## Construct the posterior as per usual.
n_iter = 5
ω̄ = rand(T)
c = zeros(T)
γ = 0.5
g = 0.5 * y_sign
β = 0
η₁_like(ω, g, β) = g .+ β .* ω
Λ_like(ω, γ) = 2 * γ * ω
@showprogress for i in 1:n_iter
    global c
    Λ = Λ_like(ω̄, γ)
    p_f = marginals(posterior(f(x, inv.(Λ)), inv.(Λ) .* η₁_like(ω̄, g, β))(x))
    @. c = sqrt(var(p_f) + abs2(mean(p_f))) / 2
    @. ω̄ = 0.5 * tanh(c) / c
end

Λ = Λ_like(ω̄, γ)
f_post = posterior(f(x, inv.(Λ)), inv.(Λ) .* η₁_like(ω̄, g, β))

# Specify some locations at which to make predictions.
T_pr = 1_200_000;
x_pr = RegularSpacing(0.0, 1e-4, T_pr);

# Compute the exact posterior marginals at `x_pr`.
f_post_marginals = marginals(f_post(x_pr));
m_post_marginals = mean.(f_post_marginals);
σ_post_marginals = std.(f_post_marginals);

# Generate a few posterior samples. Not fantastically-well optimised at present.
f_post_samples = [rand(f_post(x_pr)) for _ in 1:5];

# Visualise the posterior. The if block is just to prevent it running in CI.
# if get(ENV, "TESTING", "FALSE") == "FALSE"
using Plots
plt = plot()
# plot!(plt, x, f_true; label="", lw=2.0)
plot!(plt, f_post(x_pr); ribbon_scale=3.0, label="")
plot!(plt, x_pr, f_post_samples; color=:red, alpha=0.3, label="")
plot!(plt, x, f_true; label="", lw=2.0)
scatter!(plt, x, y; label="", markersize=1.0, alpha=1.0)

savefig(plt, "posterior.png")
# end
