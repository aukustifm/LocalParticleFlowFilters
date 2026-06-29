import Pkg; Pkg.activate(".")

# LIBRARIES ______________________________
using LinearAlgebra, JLD2, Dates, Distributions

include("../src/simulation.jl");
include("../src/blocking.jl");
include("../src/models.jl");
include("../src/filtering.jl");

# DEFINITIONS (EXPERIMENT) _____________________________________________________
K = 1000;    # Simulation horizon
Δt = 1;      # Discretisation interval

S = (60,60); # Number of grid points (Sx,Sy)
δx = 1;      # Space discretisation

# DEFINITIONS (SYSTEM) _________________________________________________________
Nx = 2*S[1]*S[2];   # State dimension after discretisation
Ny = 1*S[1]*S[2];   # Obs dimension after discretisation

## Dynamic parameters (F, k, D_u, D_v)
θₓ = [0.042, 0.06075, 1e-2, 5e-3];   # Parameter vector

## Noise parameters
Σx = 1e-4 * sqrt(Δt);         # Process noise standard deviation (already multiplied by Δt)
Σy = sqrt(1e-5);              # Measurement noise standard deviation

## Model equations (state/output)
# State equation [ dx = f(x)dt + g(x)dW ], already in discrete-time
f(x::AbstractArray{Float64}) = GrayScott_RK4(x; θ=θₓ, Δt=Δt, ∇²=∇²_FDM(S, δx));
g(x::AbstractArray{Float64}) = x*Σx; # Geometric Brownian motion

# Output equation,  y = C*x + Σy*zeta
C = [I(Nx÷2) spzeros(Nx÷2, Nx÷2)]; # Observing all states

## AUXILIARY VARIABLES__________________________________________________________
# Initial state for simulation based on a pertubation at the centre
x0 = zeros(S..., 2);
x0[:,:,1] .= 0.1randn(S...) .+ 1.0
x0[26:35,26:35,1] ./= 2; x0[26:35,26:35,2] .= 0.1randn(10,10).+0.25; 
x0 = x0[:];

# SCRIPT _________________________________

# -- Simulation / Measurement data --
_, x, y = simulate(f, g, C, Σy, x0, K, verbose=true);        # Simulates the system to generate data

# -- Filtering --
Np = 128;   # Number of particles

ssk = 6                         # Block size (per dimension)
Sb = (ssk, ssk)
Nb = (S[1]÷Sb[1])*(S[2]÷Sb[2])  # Block count
κx = BlockIndexing(Nx, S, Sb)
κy = BlockIndexing(Ny, S, Sb)

xe_1, _, _, _, t_1 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=1)
xe_2, _, _, _, t_2 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=2)
xe_3, _, _, _, t_3 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=3)
xe_4, _, _, _, t_4 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=4)
xe_5, _, _, _, t_5 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=5) # Please first set BLAS.set_num_threads(1) if Julia is already using multiple threads 
xe_6, _, _, t_6 = EnKF(f, g, C, y, Σy, x0, Np, verbose=true)
