import Pkg; Pkg.activate(".")

# LIBRARIES ______________________________
using LinearAlgebra, JLD2, Dates, Distributions

include("../utils/simulation.jl");
include("../utils/blocking.jl");
include("../models/Lorenz96.jl");
include("../utils/filtering.jl");

# DEFINITIONS (EXPERIMENT) _____________________________________________________
const Ktotal = 1099;  # Simulation horizon with initial burn-in 
const Δt = 0.01;      # Simulation horizon / Discretisation interval

tvec = Δt:Δt:(Δt*(Ktotal-99))    # Actual time horizon for filtering

xx0 = 16;   # For the covariance matrix of initial condition
     
# DEFINITIONS (SYSTEM) _________________________________________________________
Nx = 2^10       # State dimension in univariate domain
S  = (Nx,1);    # Indexing each coordinate of the state variable with a distinct natural number.

## Dynamic parameters (θ)
const θx  = 8.0; # Chaotic regime 8.0   Periodic regime 2.75

## Noise parameters
Σx = 1e0 * √(Δt);     # Process noise standard deviation (already multiplied by Δt)
Σy = 1e0 ; 

## Model equations (state/output)
# State equation [ dx = f(x)dt + g(x)dW ], already in discrete-time
f(x::AbstractArray{Float64}) = lorenz96_RK4(x; θ=θx, Δt=Δt);
g(x::AbstractArray{Float64}) = Σx; # Standard Brownian motion

# Output equation,  y = h(x) + v(x)
C  = sparse(zeros(Nx÷2, Nx)); [C[n, 2n-1] = 1.0 for n in 1:Nx÷2];
Ny = size(C, 1);

## AUXILIARY VARIABLES__________________________________________________________
# Initial state for simulation based on spatially autocorrelated variables
ρ = 0.9           # autocorrelation
σ² = 0.01         # desired variance
ϵ_dist = Normal(0, sqrt(σ² * (1 - ρ^2)))  # innovation variance for AR(1)

# Initialise and generate AR(1) process
x0 = zeros(Nx)
x0[1] = rand(Normal(θx, sqrt(σ²)))  # start from stationary distribution
for nx in 2:Nx
    x0[nx] = ρ * x0[nx-1] + rand(ϵ_dist)
end

# ________________________________________

# SCRIPT _________________________________
# -- Simulation --
_,xtotal,ytotal = simulate(f, g, C, Σy, x0, Ktotal, verbose=true);    # Simulates the system to generate data
K = Ktotal-99; 

x = xtotal[:, end-K+1:end]; y = ytotal[:, end-K+1:end]; # Removing  burn-in period
x0 .= x[:, 1]

# -- Filtering --
Np = 512;   # Number of particles

ssk = 2^8                       # Block size (per dimension)
Sb = (ssk, 1)
Nb = (S[1]÷Sb[1])*(S[2]÷Sb[2])  # Block count
κx = BlockIndexing(Nx, S, Sb)
κy = BlockIndexing(Ny, S, Sb)

xe_1, diagΣx_1, _, _, _, t_1 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=1)
xe_2, diagΣx_2, _, _, _, t_2 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=2)
xe_3, diagΣx_3, _, _, _, t_3 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=3)

#jldsave("./data/Lorenz96_F$(Int(θx))_Np$(Np)_Nx$(Nx)_timeseries_v2.jld2"; x=x, y=y, xe=(xe_1,xe_2,xe_3), diagΣx=(diagΣx_1,diagΣx_2,diagΣx_3))

ITAEₖ_1 = cumsum(Δt * tvec .* sqrt.(mean((xe_1-x).^2,dims=1))[:])
ITAEₖ_2 = cumsum(Δt * tvec .* sqrt.(mean((xe_2-x).^2,dims=1))[:])
ITAEₖ_3 = cumsum(Δt * tvec .* sqrt.(mean((xe_3-x).^2,dims=1))[:])