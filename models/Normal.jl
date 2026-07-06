# Libraries ______________
using LinearAlgebra
# ________________________

# Functions ______________
"""
    Normal(x,θ)

The Normal case .
The function returns the time-derivative 
```math
    \\frac{dx_i}{dt} = θx_i, \\quad i=1,\\dots,N_x
```
given time \$t>0\$, state \$x = [x_1\\ x_2\\ \\cdots\\ x_{N_x}]^{T}\$ and parameter \$θ\$. 
"""
function normal(x::AbstractArray{Float64}, θ::AbstractArray{Float64})
    # Dynamics
    return θ*x;
end

"""
The RK4 time-discretisation of the Lorenz 96 model.

Arguments\n
    x (1D array) : current state [concentrations U(t,s) and V(t,s)]\n
    θ (1D array) : parameters (F)
    Δt (real) : time discretisation interval
"""
@inline function normal_RK4(x::AbstractArray{Float64}; θ::AbstractArray{Float64}, Δt::Real)
    # RK4 steps
    k1 = normal(x,              θ)
    k2 = normal(x .+ k1.*Δt./2, θ)
    k3 = normal(x .+ k2.*Δt./2, θ)
    k4 = normal(x .+ k3.*Δt,    θ)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# ________________________