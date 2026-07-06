# LIBRARIES ____________________________________________________________________
using LinearAlgebra, SparseArrays

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

# Periodic
#=
function ∇²_FDM(S::Tuple{Int,Int}, Δs::Real)
    nx, ny = S
    N = nx * ny

    # Construct 2D Laplacian with periodic BCs
    function idx(i, j)
        return mod1(i, nx) + (mod1(j, ny) - 1) * nx  # wraps around both directions
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:ny
        for i in 1:nx
            center = idx(i, j)
            push!(rows, center); push!(cols, center); push!(vals, -4.0)

            # Neighbors with periodicity
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ni, nj = i + di, j + dj
                neighbor = idx(ni, nj)
                push!(rows, center)
                push!(cols, neighbor)
                push!(vals, 1.0)
            end
        end
    end

    L = sparse(rows, cols, vals) / Δs^2
    return L
end
=#

# FUNCTIONS ____________________________________________________________________
"""
The Oregonator model of the Belousov-Zhabotinsky reaction-diffusion system
The reaction network is represented by 
\$\$\\begin{aligned}
    A + Y &\\overset{k_1}{\\longrightarrow}  X + P   \\
    X + Y &\\overset{k_2}{\\longrightarrow} 2P       \\
    A + X &\\overset{k_3}{\\longrightarrow} 2X + 2Z  \\
       2X &\\overset{k_4}{\\longrightarrow}  A + P   \\
    B + Z &\\overset{k_c}{\\longrightarrow} 0.5f Y   \\
\\end{aligned}\$\$

Arguments\n
    t (float) : time [t > 0]\n
    x (1D array) : current state [concentrations X(t,s), Y(t,s) and Z(t,s)]\n
    θ (1D array) : parameters (A,B,f,κ₁,κ₂,κ₃,κ₄,kc,Dx,Dy,Dz,∇²)

Returns\n
    dx (array) : the time-derivative dx(t)/dt
"""
function Oregonator(x::AbstractVector{Float64}, θ::Vector{Float64}, ∇²::SparseMatrixCSC{Float64,Int64})
    # Parameters
    A,B,f,κ₁,κ₂,κ₃,κ₄,kc,Dx,Dy,Dz = θ;

    # Auxiliary variables
    S² = size(x,1) ÷ 3;
    X,Y,Z = (x[1:S²], x[S²+1:2S²], x[2S²+1:end])

    # Dynamics (Reaction + Diffusion)
    return [ κ₁.*A.*Y .- κ₂.*X.*Y  .+ κ₃.*A.*X .- 2κ₄.*X.^2 .+ Dx*∇²*X
            -κ₁.*A.*Y .- κ₂.*X.*Y  .+ 0.5f.*kc.*B.*Z       .+ Dy*∇²*Y
            2κ₃.*A.*X .- kc.*B.*Z                          .+ Dz*∇²*Z];
end

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
function Oregonator2(x::AbstractArray{Float64}, θ::Vector{Float64}, ∇²::SparseMatrixCSC{Float64,Int64})
    # Parameters
    ϵ,b,a,Dᵤ,Dᵥ = θ;

    # Auxiliary variables
    @views U,V = (x[1:end÷2,:], x[end÷2+1:end,:])

    # Dynamics (Reaction)
    return [1/ϵ.*(U.*(1 .- U) .- b.*V.*(U.-a)./(U.+a)) .+ Dᵤ*∇²*U;
                            U .- V                     .+ Dᵥ*∇²*V];
end

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
    k1 = Oregonator2(x,              θ, ∇²)
    k2 = Oregonator2(x .+ k1.*Δt./2, θ, ∇²)
    k3 = Oregonator2(x .+ k2.*Δt./2, θ, ∇²)
    k4 = Oregonator2(x .+ k3.*Δt,    θ, ∇²)

    # Next state
    return x .+ (k1 .+ 2k2 .+ 2k3 .+ k4).*Δt./6;
end

# ______________________________________________________________________________
