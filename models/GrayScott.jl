# Libraries ______________
using LinearAlgebra

# AUXILIARY ____________________________________________________________________

"""
Finite-different approximation (5p stencil) of the Laplacian operator

Arguments
    S  (tuple) : Number of discretised cells S = (Sₓ,Sᵧ)
    Δs (double) : State-discretisation grid size

Returns
    ∇² (array) : the linear operator approximating the Laplacian ∇²f = (∂²f/∂x₁² + ∂²f/∂x₂²)
"""
function ∇²_FDM(S::Tuple{Integer,Integer}, Δs::Real)
    ∇² =      spdiagm( 1 => repeat([2;ones(S[1]-2);0], S[2])[1:end-1]) + spdiagm(S[1] => [2ones(S[1]); ones(S[1]*(S[2]-2))]);
    ∇² = ∇² + spdiagm(-1 => repeat([0;ones(S[1]-2);2], S[2])[2:end])   + spdiagm(-S[1] => [ones(S[1]*(S[2]-2)); 2ones(S[1])]);
    ∇² = (-4I + ∇²) / Δs^2

end;

# FUNCTIONS ____________________________________________________________________
"""
Gray-Scott reaction-diffusion system

Arguments
    t (float) : time [t > 0]
    x (3D array) : current state [concentrations U(t,s) and V(t,s)]
    P (1D array) : parameters (D_u, D_v, F, k, ∇²)

Returns
    dx (array) : the time-derivative dx(t,s)/dt
"""
function GrayScott(x::AbstractArray{Float64}, P::AbstractArray{Float64}, ∇²::SparseMatrixCSC{Float64,Int64})
    # Parameters
    Dᵤ,Dᵥ,F,k = P;
    S² = size(x,1) ÷ 2;

    # Dynamics
    return [    F * (1 .- x[1:S²]) .- x[1:S²].*x[S²+1:end].^2 .+ Dᵤ*∇²*x[1:S²]    
             -(F+k) * x[S²+1:end]  .+ x[1:S²].*x[S²+1:end].^2 .+ Dᵥ*∇²*x[S²+1:end]];
end
"""
The RK4 time-discretisation of the Gray-Scott.

Arguments\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (F)
    Δt (real) : time discretisation interval
"""
@inline function GrayScott_RK4(x::AbstractArray{Float64}; P::AbstractArray{Float64}, Δt::Real,∇²::SparseMatrixCSC{Float64,Int64})
    # RK4 steps
    k1 = GrayScott(x,              P,∇²)
    k2 = GrayScott(x .+ k1.*Δt./2, P,∇²)
    k3 = GrayScott(x .+ k2.*Δt./2, P,∇²)
    k4 = GrayScott(x .+ k3.*Δt,    P,∇²)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# ________________________