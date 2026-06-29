## LIBRARIES ___________________________________________________________________
using LinearAlgebra, SparseArrays

## AUX FUNCTIONS ______________________________________________________________
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



## MODELS _____________________________________________________________________

# OREGONATOR
"""
The 2D simplified Oregonator model of the Belousov-Zhabotinsky 
reaction-diffusion system.

Arguments\n
    t (float) : time [t > 0]\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (ϵ, b,a ,Dᵤ ,Dᵥ, )
    ∇² (2D array) : 5-point stencil approximation of the Laplacian

Returns\n
    dx (array) : the time-derivative dx(t)/dt
"""
function Oregonator(x::AbstractArray{Float64}, θ::Vector{Float64}, ∇²::SparseMatrixCSC{Float64,Int64})
    # Parameters
    ϵ,b,a,Dᵤ,Dᵥ = θ;

    # Auxiliary variables
    @views U,V = (x[1:end÷2,:], x[end÷2+1:end,:])

    # Dynamics (Reaction)
    return [1/ϵ.*(U.*(1 .- U) .- b.*V.*(U.-a)./(U.+a)) .+ Dᵤ*∇²*U;
                            U .- V                     .+ Dᵥ*∇²*V];
end;

"""
The RK4 time-discretisation of the simplified Oregonator.

Arguments\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (ϵ, b,a ,Dᵤ ,Dᵥ, ∇²)
    Δt (real) : time discretisation interval
    ∇² (2D array) : 5-point stencil approximation of the Laplacian
"""
@inline function Oregonator_RK4(x::AbstractArray{Float64}; θ::Vector{Float64}, Δt::Real, ∇²::SparseMatrixCSC{Float64,Int64})
    # RK4 steps
    k1 = Oregonator(x,              θ, ∇²)
    k2 = Oregonator(x .+ k1.*Δt./2, θ, ∇²)
    k3 = Oregonator(x .+ k2.*Δt./2, θ, ∇²)
    k4 = Oregonator(x .+ k3.*Δt,    θ, ∇²)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end;

# LORENZ96
"""
    Lorenz96(x,θ)

The Lorenz 1996 chaotic system for benchmarking data assimilation.
The function returns the time-derivative 
```math
    \\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})x_{i-1} - x_i + θ, \\quad i=1,\\dots,N_x
```
given time \$t>0\$, state \$x = [x_1\\ x_2\\ \\cdots\\ x_{N_x}]^{T}\$ and parameter \$θ\$. 
The reflexive boundary conditions are defined by \$x_{-1} = x_{N_x-1}\$, \$x_{0} = x_{N_x}\$, and \$x_{N+1} = x_{1}\$.
"""
function Lorenz96(x::AbstractArray{Float64}, θ::Float64)
    # Dynamics
    #        ( xᵢ₊₁      - xᵢ₋₂        )*xᵢ₋₁         - xᵢ         + F + sum_j zᵢⱼ
    return [ ((x[2,:]     .- x[end-1,:]  ).*x[end,:]     .- x[1,:]       .+ θ)';   # (Case i = 1)  / Periodic boundary
             ((x[3,:]     .- x[end,:]    ).*x[1,:]       .- x[2,:]       .+ θ)';   # (Case i = 1)  / Periodic boundary
             (x[4:end,:]  .- x[1:end-3,:]).*x[2:end-2,:] .- x[3:end-1,:] .+ θ  ;   # (Case i = 2,…,N-1)
             ((x[1,:]     .- x[end-2,:]  ).*x[end-1,:]   .- x[end,:]     .+ θ)';   # (Case i = N)  / Periodic boundary
           ];
end

"""
The RK4 time-discretisation of the Lorenz 96 model.

Arguments\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (F)
    Δt (real) : time discretisation interval
"""
@inline function Lorenz96_RK4(x::AbstractArray{Float64}; θ::Float64, Δt::Real)
    # RK4 steps
    k1 = Lorenz96(x,              θ)
    k2 = Lorenz96(x .+ k1.*Δt./2, θ)
    k3 = Lorenz96(x .+ k2.*Δt./2, θ)
    k4 = Lorenz96(x .+ k3.*Δt,    θ)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# GAUSS-MARKOV MODEL
"""
    GaussMarkov(x,θ)

The Gauss-Markov dynamical system.
The function returns the time-derivative 
```math
    \\frac{dx_i}{dt} = θx_i, \\quad i=1,\\dots,N_x
```
given time \$t>0\$, state \$x = [x_1\\ x_2\\ \\cdots\\ x_{N_x}]^{T}\$ and parameter \$θ\$. 
"""
function GaussMarkov(x::AbstractArray{Float64}, θ::AbstractArray{Float64})
    # Dynamics
    return θ*x;
end

"""
The RK4 time-discretisation of the Gauss-Markov system.

Arguments\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (F)
    Δt (real) : time discretisation interval
"""
@inline function GaussMarkov_RK4(x::AbstractArray{Float64}; θ::AbstractArray{Float64}, Δt::Real)
    # RK4 steps
    k1 = GaussMarkov(x,              θ)
    k2 = GaussMarkov(x .+ k1.*Δt./2, θ)
    k3 = GaussMarkov(x .+ k2.*Δt./2, θ)
    k4 = GaussMarkov(x .+ k3.*Δt,    θ)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# GRAY-SCOTT MODEL
"""
Gray-Scott reaction-diffusion system

Arguments
    t (float) : time [t > 0]
    x (3D array) : current state [concentrations U(t,s) and V(t,s)]
    P (1D array) : parameters (D_u, D_v, F, k, ∇²)

Returns
    dx (array) : the time-derivative dx(t,s)/dt
"""
function GrayScott(x::AbstractArray{Float64}, θ::AbstractArray{Float64}, ∇²::SparseMatrixCSC{Float64,Int64})
    # Parameters
    f,k,Dᵤ,Dᵥ = θ;

    # Auxiliary variables
    @views U,V = (x[1:end÷2,:], x[end÷2+1:end,:])

    # Dynamics
    return [ f*(1.0.-U) - U.*V.^2 .+ Dᵤ*∇²*U
             -(f+k)*V   + U.*V.^2 .+ Dᵥ*∇²*V];
end
"""
The RK4 time-discretisation of the Gray-Scott.

Arguments\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (F)
    Δt (real) : time discretisation interval
"""
@inline function GrayScott_RK4(x::AbstractArray{Float64}; θ::AbstractArray{Float64}, Δt::Real,∇²::SparseMatrixCSC{Float64,Int64})
    # RK4 steps
    k1 = GrayScott(x,              θ, ∇²)
    k2 = GrayScott(x .+ k1.*Δt./2, θ, ∇²)
    k3 = GrayScott(x .+ k2.*Δt./2, θ, ∇²)
    k4 = GrayScott(x .+ k3.*Δt,    θ, ∇²)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# ______________________________________________________________________________

