import Pkg; Pkg.activate(".")

# LIBRARIES ______________________________
using LinearAlgebra, JLD2, Dates, Distributions

include("../utils/simulation.jl");
include("../utils/blocking.jl");
include("../models/GrayScott.jl");
include("../utils/filtering.jl");

# DEFINITIONS (EXPERIMENT) _____________________________________________________
K = 1000;       # Simulation horizon
Δt = 1;      # Discretisation interval

S = (60,60);    # Number of grid points (Sx,Sy)
δx = 1;      # Space discretisation

# DEFINITIONS (SYSTEM) _________________________________________________________
Nx = 2*S[1]*S[2];   # State dimension after discretisation
Ny = 10*S[1]*S[2];  # Obs dimension after discretisation

## Dynamic parameters (D_u, D_v, F, k, ∇²)
Pₓ  = [2e-5, 1e-5, 0.042, 0.06075];   # Parameter vector

## Noise parameters
Σx = 1e-8 * √(Δt);     # Process noise standard deviation (already multiplied by Δt)
Σy = sqrt(1e-5);             # Measurement noise standard deviation

## Model equations (state/output)
# State equation [ dx = f(x)dt + g(x)dW ], already in discrete-time
f(x::AbstractArray{Float64}) = GrayScott_RK4(x; P=Pₓ, Δt=Δt, ∇²=∇²_FDM(S, δx));
g(x::AbstractArray{Float64}) = x*Σx; # Geometric Brownian motion

# Output equation,  y = C*x + Σy*zeta
C = sparse(Matrix{Float64}(I(Nx))) # Observing all states
C = C[1:2:Nx,:]                    # Observing every other state

## AUXILIARY VARIABLES__________________________________________________________
# Initial state for simulation based on a pertubation at the centre
x0 = zeros(S...,2);
x0[:,:,1] .= 0.1randn(S...) .+ 1.0
x0[26:35,26:35,1] ./= 2; x0[26:35,26:35,2] .= 0.1randn(10,10).+0.25; 

# SCRIPT _________________________________

# -- Simulation / Measurement data --
_, x, y = simulate(f, g, C, Σy, x0[:], K, verbose=true);        # Simulates the system to generate data

# -- Filtering --
Np = 128;   # Number of particles

SNR = hcat( [norm(f(xx)*Δt)./norm(g(xx)*√(Δt).*randn(Nx)) for xx in eachcol(x)]... );

for pp in [4 6 8]
    Nₚ = 2^pp;                              # Number of particles 
    πₓ₀ = MvNormal(x₀[:], 1e-4I(Nx));       # Initial filtering distribution

    μₑ, _, _ = BlockPF_Eff((F̃,C,10Σₓ,Σᵧ), y, πₓ₀, Nₚ, S, Sₖ=5, verbose=true);
    μₑ, _, _ = DH((F̃,C,10Σₓ,Σᵧ), y[:, 1:2], πₓ₀, Nₚ, verbose=true);

    RMSE = sqrt.(mean((x-μₑ).^2, dims=1))
    jldsave("../data/GreyScott_Small_Inflated_StrongX0_Sk5_N6_P$(pp)_Ny2.jld2"; SNR=SNR, RMSE=RMSE, x=x.s, y=y.s, μₑ=μₑ.s, Σₓ=Σₓ, Σᵧ=Σᵧ)
end

# PLOTTING ______________________________

using Plots, LaTeXStrings, Measures
S² = prod(S)
c_max = ((0, maximum(x[1:S²,:])), (0, maximum(x[S²+1:2S²,:])));
gif = @gif for k = 1:10:size(x,2)
    XX  = collect(get_grids(x[:,k],S));
    XXₑ = collect(get_grids(μₑ[:,k],S));
    EE  = (sqrt.((XX[1]-XXₑ[1]).^2), sqrt.((XX[2]-XXₑ[2]).^2))

    ht = (heatmap(EE[1] , title="Error ($(round(mean(EE[1]), digits=2)), $(round(mean(EE[2]), digits=2)))"   , xlabel=""      , ylabel=L"$x_2$", colorbar_title="",     c=:berlin),
          heatmap(XX[1] , title="State"   , xlabel=""      , ylabel=L"$x_2$", clims=c_max[1], colorbar_title="",     c=:berlin),      
          heatmap(XXₑ[1], title="Estimate", xlabel=""      , ylabel=""      , clims=c_max[1], colorbar_title=L"$U$", c=:berlin))
    hd = (heatmap(EE[2] , title=""        , xlabel=L"$x_1$", ylabel=L"$x_2$", clims=c_max[2], colorbar_title="",     c=:berlin),
          heatmap(XX[2] , title=""        , xlabel=L"$x_1$", ylabel=L"$x_2$", clims=c_max[2], colorbar_title="",     c=:berlin),
          heatmap(XXₑ[2], title=""        , xlabel=L"$x_1$", ylabel=""      , clims=c_max[2], colorbar_title=L"$V$", c=:berlin));

    plot(ht..., hd..., layout=(2,3), size=(900,400), margin=2mm, guidefontsize=24, colorbar_titlefontsize=16, 
                    colorbar=:best, ticks=false, bbox_inches="tight")
end
# ________________________________________
