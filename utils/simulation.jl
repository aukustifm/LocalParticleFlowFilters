# Libraries ___________________________________________________________________
using LinearAlgebra, SparseArrays, ProgressMeter, Random
#using MKLSparse     # This forces a wrapper to the Intel's MKL SparseSuite, making sparse-dense operations multi-threaded
# ______________________________________________________________________________

# Functions ____________________________________________________________________
"""
Simulates a dynamical system equipped with a measurement process

Arguments\n
    f (Function) : state-to-state equation from dx = f(x)dt + g(x)dw
    g (Function) : noise-to-state equation from dx = f(x)dt + g(x)dw
    C (SparseMatrixCSC) : measurement emission matrix
    Σy (UniformScaling) : measurement noise covariance matrix (isotropic)
    x0 (1D array) : initial state
    K (int) : number of iterations
    verbose (bool) : Enables/disable the progress bar (default: false)
"""
function simulate(f::Function,
                  g::Function,
                  C::SparseMatrixCSC{Float64,Int64},
                  Σy::Float64,
                  x0::Vector{Float64}, 
                  K::Integer; 
                  verbose::Bool=false)

    # Auxiliary variables
    p_bar = Progress(K-1, desc="Simulation:", dt=0.5, barlen=50, showspeed=true, color=:white);

    Ny,Nx = size(C);
    x = Array{Float64}(undef, Nx, K)

    # Simulation loop
    x[:,1] .= x0; 
    for k = 1:Int(K-1); if verbose; next!(p_bar); end;
        x[:,k+1] .= f(view(x,:,k)) .+ g(view(x,:,k)).*randn(Nx)
    end

    y = C*x + Σy.*randn(Ny,K);

    # -- --
    return (1:K), x, y
end;

# ______________________________________________________________________________
