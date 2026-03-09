import Pkg; Pkg.activate(".")

# LIBRARIES ______________________________
using LinearAlgebra, JLD2, Dates, Distributions

include("../utils/simulation.jl");
include("../utils/blocking.jl");
include("../models/Normal.jl");
include("../utils/filtering.jl");

# DEFINITIONS (EXPERIMENT) _____________________________________________________
K = 50; 
Δt = 0.01;      # Simulation horizon / Discretisation interval

# Model parameters
function generate_stable_matrix(dim::Int, lambda_val::Float64)
    # Random eigenvalues with negative real parts
    real_parts = -rand(Exponential(lambda_val), dim)
    # All eigenvalues are real for simplicity
    eigenvalues = Diagonal(real_parts)

    # Random orthogonal matrix
    random_matrix = randn(dim, dim)
    Q, _ = qr(random_matrix) # Q is orthogonal

    # A stable matrix A
    A = Q * eigenvalues * Q'

    return A
end

# Multiple runs 
logssk_vec = [10,8,6,4]
Nb_diff = length(logssk_vec) # Number of different block sizes (upper bound)
Ns = 1     # Number of simulations

#ITWDₖ_1 = [zeros(Nb_diff) for _ in 1:Ns]
#ITWDₖ_2 = [zeros(Nb_diff) for _ in 1:Ns]
#ITWDₖ_3 = [zeros(Nb_diff) for _ in 1:Ns]
#ITAEₖ_4 = [zeros(Nx_diff,Nb_diff) for _ in 1:Ns]

#tₖ_1 = [zeros(Nb_diff) for _ in 1:Ns]
#tₖ_2 = [zeros(Nb_diff) for _ in 1:Ns]
#tₖ_3 = [zeros(Nb_diff) for _ in 1:Ns]
#tₖ_4 = [zeros(Nx_diff,Nb_diff) for _ in 1:Ns]
## Dynamic parameters (θ)
θx = generate_stable_matrix(Nx, 10.0)   # convert to dense Matrix{Float64} if you need

Σx = (1 * √(Δt));     # Process noise standard deviation (already multiplied by Δt)
Σy = (1 * √(Δt)) ;    # Measurement noise standard deviation

## Model equations (state/output)
# State equation [ dx = f(x)dt + g(x)dW ], already in discrete-time
f(x::AbstractArray{Float64}) = normal_RK4(x; θ=θx, Δt=Δt);
g(x::AbstractArray{Float64}) = Σx; # Standard Brownian motion

# Output equation,  y = Cx + v(x)
Nx = 1024
C = sparse(Matrix{Float64}(I(Nx))) # Observing all states
C = C[1:2:Nx,:]                    # Observing every other state

## AUXILIARY VARIABLES__________________________________________________________

Ny = size(C, 1);
S  = (Nx,1);  

x0 = randn(Nx); # Initial state for simulation
    
# ________________________________________

# SCRIPT _________________________________
# -- Simulation

_,x, y = simulate(f, g, C, Σy, x0, K, verbose=true);    # Simulates the system to generate data


# -- Ground truth via Kalman Filter

μx = Array{Float64,2}(undef, Nx, K );      # Matrix of particle sample-mean
μx[:,1] = x0;
Σopt = [sparse(zeros(Nx, Nx)) for _ in 1:K] 
F = exp(θx*Δt)

# Continuous-discrete Kalman filter loop
println("Obtaining the Kalman filter solution")
for k = 2:Int(K)
    
    # Prediction step (only process noise affects uncertainty)
    μx[:,k] = f(μx[:,k-1])
    Σopt[k] .= F*Σopt[k-1]*F' .+ Σx^2*I(Nx)
        
    # Kalman gain
    Innovation_covariance = C*Σopt[k]*C' .+ Σy^2*I(Ny)
    Gain = Σopt[k] * C' / Matrix(Innovation_covariance) # Check conjugate gradient (LinearSolvers.jl CG())

    # Update estimate and covariance
    μx[:,k] += Gain * (view(y,:,k) - C*μx[:,k])
    Σopt[k] = (I(Nx) - Gain*C) * Σopt[k] * (I(Nx) - Gain*C)' + Gain*Σy^2*I(Ny)*Gain'
end

jldsave("./res/MvNormal_Nx$(Nx)_KFsolution.jld2"; x=x, y=y, μx=μx, Σopt=Σopt, θx=θx)

# -- Filtering

Np = 1024;

WD = zeros(3,Nb_diff) # Storing wasserstein2 distance
Xp_marginal = [zeros(Nb_diff, Nx, Np) for _ in 1:3] 

Ns = 20 # Number of runs

for ns = 1:Ns
    for logssk_id in axes(logssk_vec,1) 

        ssk = 2^logssk_vec[logssk_id];              # Block size
        Sb = (ssk, 1)
        Nb = (S[1]÷Sb[1])*(S[2]÷Sb[2])
        κx = BlockIndexing(Nx, S, Sb)
        κy = BlockIndexing(Ny, S, Sb)

        xe_1, Σx_1, Xp_1, _, _, t_1 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=1)
        xe_2, Σx_2, Xp_2, _, _, t_2 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=2)
        xe_3, Σx_3, Xp_3, _, _, t_3 = BlockPF(f, g, C, y, Σy, x0, Np, κx, κy, verbose=true, bpf_type=3)

        tvec = range(Δt, K*Δt, length=K);

        jldsave("./res/MvNormal_Np$(Np)_Nx$(Nx)_ssk$(ssk)_timeseries_ns$(ns).jld2"; xe=(xe_1,xe_2,xe_3), Σx = (Σx_1,Σx_2,Σx_3), Xp=(Xp_1,Xp_2,Xp_3))
        
        WD[:,logssk_id] = [
        wasserstein2(μx[:, end], Σopt[end], xe_1[:, end], Σx_1[end]),
        wasserstein2(μx[:, end], Σopt[end], xe_2[:, end], Matrix(Σx_2[end])),
        wasserstein2(μx[:, end], Σopt[end], xe_3[:, end], Matrix(Σx_3[end]))]

        Xp_marginal[1][logssk_id, :, :] = Xp_1
        Xp_marginal[2][logssk_id, :, :] = Xp_2
        Xp_marginal[3][logssk_id, :, :] = Xp_3

        println("wasserstein2 dist: ", WD[:,logssk_id])

    end
end

#jldsave("./res/MvNormal_Np$(Np)_Nx$(Nx)_WD.jld2"; results=(μx[end], Σopt[end], Xp_marginal, WD)) 
#dt = load("./res/MvNormal_Np$(Np)_Nx$(Nx)_WD.jld2") 
#μx[end], Σopt[end], Xp_marginal, WD = dt["results"]

WDnorm = WD./maximum(WD)

# -- Visualising the results for an arbitrary coordinate
nx = 100

using Plots, LaTeXStrings, CairoMakie
col_pal = distinguishable_colors(30, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)[[27, 3, 4, 2]]

set_theme!(fonts=(; regular="New Century Schoolbook Roman"))

dc_font = 25  
dc_width = 1500
dc_height = 400

using Plots, KernelDensity;

lb, ub = μx[nx,end]-4sqrt(Σopt[end][nx,nx]), μx[nx,end]+5.75sqrt(Σopt[end][nx,nx]);
xrange = lb:0.01:ub

begin
    ff = CairoMakie.Figure(size = (dc_width, dc_height), dpi=300,
    fontsize = dc_font, font="New Century Schoolbook Roman")

    ax1 = ff[1,1]

    ax11 = ax1[1, 1] = Axis(ff,
        xgridvisible = false,
        ygridvisible = false,
        limits=(-1.4,1.75, 0,4.25),
        yticks             = [0,2,4],
        yminorticks        = IntervalsBetween(3),
        backgroundcolor = :white;)
    ax12 = ax1[1, 2] = Axis(ff,
        xgridvisible = false,
        ygridvisible = false,
        yticklabelsvisible = false,
        limits=(-1.4,1.75, 0, 4.25),
        yticks             = [0,2,4],
        yminorticks        = IntervalsBetween(3),
        backgroundcolor = :white; )
    ax13 = ax1[1, 3] = Axis(ff,
        xgridvisible = false,
        ygridvisible = false,
        yticklabelsvisible = false,
        limits=(-1.4,1.75,  0, 4.25),
        yticks             = [0,2,4],
        yminorticks        = IntervalsBetween(3),
        backgroundcolor = :white;)
    ax14 = ax1[1, 4] = Axis(ff,
        xgridvisible = false,
        ygridvisible = false,
        yticklabelsvisible = false,
        limits=(-1.4,1.75,  0, 4.25),
        yticks             = [0,2,4],
        yminorticks        = IntervalsBetween(3),
        backgroundcolor = :white; )
        

    for i in 1:4
        lines!(eval(Symbol("ax1".*string(i))), xrange, z -> pdf(Normal(μx[nx,end], sqrt(Σopt[end][nx,nx])), z),
        linewidth=2, color=col_pal[1])

        label_text = "W = $(round(WDnorm[1, i], digits=2))"
        lines!(eval(Symbol("ax1".*string(i))), xrange, z -> pdf(kde(Xp_marginal[1][i,nx,:]), z),
            label = label_text,
            linewidth=2, color=col_pal[2])

        label_text = "W = $(round(WDnorm[2, i], digits=2))"
        lines!(eval(Symbol("ax1".*string(i))), xrange, z -> pdf(kde(Xp_marginal[2][i,nx,:]), z),
            label = label_text,
            linewidth=2, color=col_pal[3])

        label_text = "W = $(round(WDnorm[3, i], digits=2))"
        lines!(eval(Symbol("ax1".*string(i))), xrange, z -> pdf(kde(Xp_marginal[3][i,nx,:]), z),
            label = label_text,
            linewidth=2, color=col_pal[4])
  
    end
    axislegend(ax11, L"\log_2\overline{V}_b = 10"; position = :rt, framevisible = false, labelsize = dc_font, patchsize = (18,8))
    axislegend(ax12, L"\log_2\overline{V}_b = 8"; position = :rt, framevisible = false, labelsize = dc_font, patchsize = (18,8))
    axislegend(ax13, L"\log_2\overline{V}_b = 6"; position = :rt, framevisible = false, labelsize = dc_font, patchsize = (18,8))
    axislegend(ax14, L"\log_2\overline{V}_b = 4"; position = :rt, framevisible = false, labelsize = dc_font, patchsize = (18,8))

   
    elems = [LineElement(color=col, linewidth=2) for col in col_pal]
    legend = Legend(ff, elems, ["Kalman filter", "locSIR-std", "locSIR-opt", "locFPF"];
        orientation = :horizontal, labelsize = dc_font)

    # Place legend on top of the figure
    ff[0, :] = legend

end

ff

save("./res/MvNormal_Nx1024.png", ff)
save("./res/MvNormal_Nx1024.pdf", ff)