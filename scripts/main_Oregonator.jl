import Pkg; Pkg.activate(".")

# LIBRARIES ____________________________________________________________________
using LinearAlgebra, JLD2, Dates

include("../utils/simulation.jl");
include("../utils/blocking.jl");
include("../models/Oregonator.jl");
include("../utils/filtering.jl");

# AUXILIARY FUNCTIONS __________________________________________________________
⊗(A,B) = kron(sparse(A), B);                # Alias for sparse Kronecker product

Φᵤ(u) = exp.(-1/(2*15) .* (u.-10).^2)       # Spectral components of U
Φᵥ(v) = exp.(-1/(2*15) .* (v.-40).^2)       # Spectral components of V

# DEFINITIONS (EXPERIMENT) _____________________________________________________
K = 3000;       # Simulation horizon
Δt = 0.01;      # Discretisation interval

S = (80,80);    # Number of grid points (Sx,Sy)
δx = 0.02;      # Space discretisation

# DEFINITIONS (SYSTEM) _________________________________________________________
Nx = 2*S[1]*S[2];   # State dimension after discretisation
Ny = 10*S[1]*S[2];  # Obs dimension after discretisation

## Dynamic parameters (ϵ, b, q, Du, Dv)
θₓ = [0.08, 0.95, 0.0075, 5e-4, 5e-6];     # https://cattech-lab.com/science-tools/bz-reaction/

## Noise parameters
Σx = 1e-2 * √(Δt);     # Process noise standard deviation (already multiplied by Δt)
Σy = sqrt(1e-5);             # Measurement noise standard deviation

## Model equations (state/output)
# State equation [ dx = f(x)dt + g(x)dW ], already in discrete-time
f(x::AbstractArray{Float64}) = Oregonator_RK4(x; θ=θₓ, Δt=Δt, ∇²=∇²_FDM(S, δx));
g(x::AbstractArray{Float64}) = x*Σx; # Geometric Brownian motion

# Output equation,  y = C*x + Σy*zeta
C = [Φᵤ.(0:5:50-1) ⊗ I(S[1]*S[2])  Φᵥ.(0:5:50-1) ⊗ I(S[1]*S[2])]    # Output matrix

## AUXILIARY VARIABLES__________________________________________________________
# Initial state for simulation based on a pertubation of the steady-state solution
ϵ,α,q = θₓ[1:3];
x0 = ones(S[1]*S[2]*2) .* (1 - α - q + sqrt((α+q-1)^2 + 4q*(1+α))) / 2;

# ______________________________________________________________________________
# SCRIPT _______________________________________________________________________

# -- Simulation / Measurement data --
_, x, y = simulate(f, g, C, Σy, x0, K, verbose=true);        # Simulates the system to generate data

# -- Filtering --
Np = 128;   # Number of particles

ssk = 8                         # Block size (per dimension)
Sb = (ssk, ssk)
Nb = (S[1]÷Sb[1])*(S[2]÷Sb[2])  # Block count
κx = BlockIndexing(Nx, S, Sb)
κy = BlockIndexing(Ny, S, Sb)

xe_1, _, _, _, t_1 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=1)
xe_2, _, _, _, t_2 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=2)
xe_3, _, _, _, t_3 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=3)

jldsave("./data/Oregonator_Np$(Np)_Nx$(Nx)_ssk$(ssk).jld2"; elapsed=(tₖ_1[ssk_id], tₖ_2[ssk_id],tₖ_3[ssk_id]), RMSE=(RMSEₖ_1[ssk_id],RMSEₖ_2[ssk_id],RMSEₖ_3[ssk_id]))

# Actual time horizon for filtering
tvec = range(Δt, K*Δt, length=K) 

ITAEₖ_1 = cumsum(Δt * tvec .* sqrt.(mean((xe_1-x).^2,dims=1))[:])
ITAEₖ_2 = cumsum(Δt * tvec .* sqrt.(mean((xe_2-x).^2,dims=1))[:])
ITAEₖ_3 = cumsum(Δt * tvec .* sqrt.(mean((xe_3-x).^2,dims=1))[:])