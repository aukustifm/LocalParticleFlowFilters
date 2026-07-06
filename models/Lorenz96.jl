# Libraries ______________
using LinearAlgebra
# ________________________

# Functions ______________
"""
    lorenz96(x,θ)

The Lorenz 1996 chaotic system for benchmarking data assimilation.
The function returns the time-derivative 
```math
    \\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})x_{i-1} - x_i + θ, \\quad i=1,\\dots,N_x
```
given time \$t>0\$, state \$x = [x_1\\ x_2\\ \\cdots\\ x_{N_x}]^{T}\$ and parameter \$θ\$. 
The reflexive boundary conditions are defined by \$x_{-1} = x_{N_x-1}\$, \$x_{0} = x_{N_x}\$, and \$x_{N+1} = x_{1}\$.
"""
function lorenz96(x::AbstractArray{Float64}, θ::Float64)
    # Dynamics
    #        ( xᵢ₊₁      - xᵢ₋₂        )*xᵢ₋₁         - xᵢ         + F + sum_j zᵢⱼ
    return [ ((x[2,:]     .- x[end-1,:]  ).*x[end,:]     .- x[1,:]       .+ θ)';   # (Case i = 1)  / Periodic boundary
             ((x[3,:]     .- x[end,:]    ).*x[1,:]       .- x[2,:]       .+ θ)';   # (Case i = 1)  / Periodic boundary
             (x[4:end,:] .- x[1:end-3,:]).*x[2:end-2,:] .- x[3:end-1,:] .+ θ;   # (Case i = 2,…,N-1)
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
@inline function lorenz96_RK4(x::AbstractArray{Float64}; θ::Float64, Δt::Real)
    # RK4 steps
    k1 = lorenz96(x,              θ)
    k2 = lorenz96(x .+ k1.*Δt./2, θ)
    k3 = lorenz96(x .+ k2.*Δt./2, θ)
    k4 = lorenz96(x .+ k3.*Δt,    θ)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# ________________________