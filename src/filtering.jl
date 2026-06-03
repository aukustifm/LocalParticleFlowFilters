# LIBRARIES ____________________________________________________________________
using LinearAlgebra, Statistics, Distributions, SharedArrays, Distances, DifferentialEquations
include("sampling.jl")

# AUXILIARY FUNCTIONS __________________________________________________________
makeProgressBar(K, title) = Progress(K, desc=title, dt=0.5, barlen=50, showspeed=true, color=:white);
PF_ProgressSummary(k, ηp) = [(:k,k), (:ηₚ, (extrema(ηp[:,k-1])..., round(mean(ηp[:,k-1]))))]

# PARTICLE FILTERS _____________________________________________________________
"""
    BlockPF(.)

    Block particle filter for isotropic noise variances.
"""
function BlockPF(f::Function,
                 g::Function,
                 C::SparseMatrixCSC{Float64,Int64},
                 y::Matrix{Float64}, 
                 Σy::Float64,
                 x0::Vector{Float64}, 
                 Np::Integer,
                 κx::Matrix{Int64},
                 κy::Matrix{Int64};
                 init_d::Bool = false,
                 verbose::Bool = false,
                 bpf_type::Int64 = 1)

    # Computes dimensions
    Ny, K = size(y);                            # Dimension of output vector / Number of measurements
    Nx = length(x0);                            # Dimension of state vector
    Nb = size(κx,2);                            # Number of blocks

    ## Initialize the state, particle and weight sequences
    Xp = Array{Float64,2}(undef, Nx, Np);      # Matrix of particles
    Wp = Array{Float64,2}(undef, Nx, Np);      # Matrix of sampled noise for jittering each particle
    dy = Array{Float64,2}(undef, Ny, Np);      # Matrix of measurement deviations for particles
    z  = Array{Float64,2}(undef, Nb, Np);      # Matrix of normalizing constants for the weights (per-block)
    μx = Array{Float64,2}(undef, Nx, K);      # Matrix of particle sample-mean
    diagΣx = Array{Float64,2}(undef, Nx, K); diagΣx[:, 1] .= 0.0       # Matrix of particle sample-mean
    # If running Normal-Normal case:
    # Σx = [spzeros(Nx,Nx) for _ in 1:K] # Matrix of particle sample-mean
    ω  = Array{Float64,2}(undef, Np, Nb);      # Matrix of importance weights (per-block)
    ηp = Array{Float64,2}(undef, Nb, K );      # Matrix of effective number of particles (per-block)

    Xp = repeat(x0, 1, Np); 
    if init_d  
        Xp = randn(Nx,Np)
        # Σx[1] = I(Nx)
        diagΣx[:,1] .= 1.0
    end
    μx[:,1] = x0; 
    ω[:] .= 1 ./ Np;

    ## PRE-PROCESSING: Auxiliary variables, recycle variables, scaling, etc.
    C .= C ./ Σy;    # Scales the emission matrix and measurement data 
    y .= y ./ Σy;    #  so that computations are equivalent to Σy = I

    Σopt = [spzeros(Nx,Nx) for _ in axes(Xp,2)] # Vector of optimal covariance matrices (for performance reasons)
    C11::Vector{Float64} = diag(C[:,1:end÷2]'C[:,1:end÷2])
    C12::Vector{Float64} = diag(C[:,1+end÷2:end]'C[:,1:end÷2])
    C22::Vector{Float64} = diag(C[:,1+end÷2:end]'C[:,1+end÷2:end])

    ## FILTERING LOOP: Iterates over all measurements
    p = makeProgressBar(K-1, "Filtering (Type: $(bpf_type)): Nx=$(Nx), Nb=$(Nb), Np=$(Np), K=$(K))")
    for k = 2:Int(K); if verbose; next!(p, showvalues=PF_ProgressSummary(k,ηp)); end;
        ParticleResample!(Xp, ω, bpf_type, κx)                                      # Resample from prior distribution
        ParticlePropagate!(Xp, Wp, Σopt, z, dy, view(y,:,k), f, g, C, C11, C12, C22, Σy, bpf_type, κx, κy)   # Sample new ensemble (by propagating w/ dynamics)
        ParticleWeights!(ω, z, dy, bpf_type, κy)                                        # Compute importance weights
        ParticleEstimate!(k, μx, Xp, ω, bpf_type, κx)                                   # Compute the current estimate

        # METRICS: Effective number of particles
        ηp[:,k] = round.(1 ./ sum(ω.^2, dims=1));
        
        # COV: For the Gaussian case! Mute this otherwise.
        diagΣx[:,k] = var(Xp, dims=2)
        #Σx[k] = cov(Xp')

    end
    
    # POST-PROCESSING: Cleanup, re-scaling, ...
    C .= Σy .* C;    # Rescales the emission matrix and measurement data 
    y .= Σy .* y;    #  back to the original magnitudes

    # -- --
    #return μx, Σx, Xp, ω, ηp, [p.tinit, p.tsecond, p.tlast] 
    return μx, diagΣx, Xp, ω, ηp, [p.tinit, p.tsecond, p.tlast] 
end;

# PARTICLE OPERATIONS _________________________
function ParticleResample!(Xp::Matrix{Float64}, ω::Matrix{Float64}, bpf_type::Int64, κx::Matrix{Int64})
    if bpf_type in (3,4); return; end   # Does not resample if Block Particle Flow
    
    Threads.@threads for b in axes(κx,2);
        Xp[κx[:,b],:] .= Xp[κx[:,b], SystematicResampling(ω[:,b])];
    end
end

function ParticlePropagate!(Xp::Matrix{Float64}, 
                            Wp::Matrix{Float64}, 
                            Σopt::Vector{SparseMatrixCSC{Float64,Int64}}, 
                            z::Matrix{Float64}, 
                            dy::Matrix{Float64}, 
                            y::AbstractArray{Float64}, 
                            f::Function, 
                            g::Function, 
                            C::SparseMatrixCSC{Float64,Int64}, 
                            C11::Vector{Float64},
                            C12::Vector{Float64},
                            C22::Vector{Float64},
                            Σy::Float64, 
                            bpf_type::Int64,
                            κx::Matrix{Int64},
                            κy::Matrix{Int64})

    # Pre-computes the noise-matrix and propagates the prior mean
    Wp .= g(Xp);
    Xp .= f(Xp);

    # Block particle filter w/ standard importance distribution
    if bpf_type == 1
        Xp .= Xp .+ Wp.*randn(size(Xp));
        z  .= 0
        dy .= y .- C*Xp;

    # Block particle filter w/ optimal importance distribution
    elseif bpf_type == 2
        dy .= y .- C*Xp;

        ParticleFastInverse!(Σopt, Wp, C11, C12, C22)
        ParticleUpdate!(Xp, Wp, C, y, Σopt)
        ParticleWeightsAux!(dy, z, C, Σopt, κx, κy)

    # Block particle filter w/ optimal importance distribution and particle flow
    elseif bpf_type == 3
        Xp .= Xp .+ Wp.*randn(size(Xp));
        
        for b in axes(κx,2)
            #θ_PF = (C[κy[:,b],κx[:,b]]'y[κy[:,b]], C[κy[:,b],κx[:,b]]'C[κy[:,b],κx[:,b]], getindex.(Σopt,[κx[:,b]],[κx[:,b]]))
            θ_PF = (C[κy[:,b],κx[:,b]]'y[κy[:,b]], C[κy[:,b],κx[:,b]])
            Xp[κx[:,b],:] .= solve(ODEProblem(ParticleFlux!, view(Xp,κx[:,b],:), (0.0,1.0), θ_PF), save_everystep=false)(1.0) #progress=true, 
        end
        
    elseif bpf_type == 4
        Xp .= Xp .+ Wp.*randn(size(Xp));
        
        PCt = Array{Float64}(undef, Nx, Ny);
        A = Array{Float64}(undef, Nx, Nx); 
        for b in axes(κx,2)    
            θ_PF = (A[κx[:,b],κx[:,b]], y[κy[:,b]], C[κy[:,b],κx[:,b]], PCt[κx[:,b],κy[:,b]])
            Xp[κx[:,b],:] .= solve(ODEProblem(ParticleExactFlux!, view(Xp,κx[:,b],:), (0.0,1.0), θ_PF), progress=true, save_everystep=false)(1.0) #,
        end
    end
end

function ParticleWeights!(ω::Matrix{Float64}, z::Matrix{Float64}, dy::Matrix{Float64}, bpf_type::Int64, κy::Matrix{Int64})
    if bpf_type in (3,4); return; end   # Does not resample if Block Particle Flow

    Threads.@threads for b in axes(ω,2)
        ω[:,b] = [-dot(dy_b,dy_b) for dy_b in eachcol(view(dy,κy[:,b],:))];         # This is the Gaussian kernel ω = -0.5dy'dy, but using efficient numerics
    end

    ω .= ω .+ z';
    ω .= exp.(ω .- maximum(ω, dims=1));  
    ω .= ω ./ sum(ω, dims=1)
end

function ParticleEstimate!(k::Int64, μx::Matrix{Float64}, Xp::Matrix{Float64}, ω::Matrix{Float64}, bpf_type::Int64,  κx::Matrix{Int64})
    Threads.@threads for b in axes(ω,2)
        μx[κx[:,b],k] .= Xp[κx[:,b],:] * ω[:,b];
    end
end

function ParticleFastInverse!(Σopt::Vector{SparseMatrixCSC{Float64,Int64}}, Wp::Matrix{Float64}, C11::Vector{Float64}, C12::Vector{Float64}, C22::Vector{Float64})
    Threads.@threads for p in axes(Wp,2)
        # For ease of notation
        D_inv = @. 1 / sqrt(C22 + 1/Wp[1+end÷2:end,p]^2)
        Σ_Schur = @. 1 / sqrt(C11 + 1/Wp[1:end÷2,p]^2 - C12^2 * D_inv)

        # Cholesky factorization of the inverse Σopt = (inv(Σb) + C'inv(Σy)C)
        Σopt[p][1:end÷2,    1:end÷2    ] = Diagonal(Σ_Schur)
        Σopt[p][1+end÷2:end,1:end÷2    ] = Diagonal(-D_inv .* C12 .* Σ_Schur)
        Σopt[p][1+end÷2:end,1+end÷2:end] = Diagonal(D_inv)
    end
end

function ParticleUpdate!(Xp::Matrix{Float64}, Wp::Matrix{Float64}, C::SparseMatrixCSC{Float64,Int64}, y::AbstractVector{Float64}, Σopt::Vector{SparseMatrixCSC{Float64,Int64}})
    Xp .= Xp ./ Wp.^2 .+ C'y
    Threads.@threads for p in axes(Xp,2)
        Xp[:,p] .= Σopt[p]*(Σopt[p]'Xp[:,p] .+ randn(size(Xp,1)))
    end
end

function ParticleWeightsAux!(dy::Matrix{Float64}, z::Matrix{Float64}, C::SparseMatrixCSC{Float64,Int64}, Σopt::Vector{SparseMatrixCSC{Float64,Int64}}, κx::Matrix{Int64}, κy::Matrix{Int64})
    Threads.@threads for p in axes(dy,2)
        Σ_yx = C*Σopt[p]
        for b in axes(κy,2)
            kxb,kyb = (κx[:,b], κy[:,b])
            z[b,p] = logdet(I - Σ_yx[kyb,kxb]'Σ_yx[kyb,kxb]) + dot(Σ_yx[kyb,kxb]'dy[kyb,p], Σ_yx[kyb,kxb]'dy[kyb,p]) 
        end
    end
end

function ParticleFlux!(du::AbstractMatrix{Float64}, u::AbstractMatrix{Float64}, p::Tuple{AbstractVector{Float64},SparseMatrixCSC{Float64,Int64}}, λ::Float64)#,Vector{SparseMatrixCSC{Float64,Int64}}
    #=
    y,CtC,Σopt = p
    Σλ = cov(u, dims=2); 
    e = y .- 0.5*CtC*(u .+ mean(u, dims=2));

    Threads.@threads for p in axes(u,2)
        du[:,p] .= Σλ*e[:,p] .- Σλ*CtC*Σopt[p]*Σopt[p]'e[:,p]
    end
    =#
    y, C = p
    Σλ = cov(u, dims=2); 
    e = y .- 0.5*C'C*(u .+ mean(u, dims=2));

    Threads.@threads for p in axes(u,2)
        du[:,p] .= Σλ*e[:,p]
    end


end

function ParticleExactFlux!(du::AbstractMatrix{Float64}, u::AbstractMatrix{Float64}, p::Tuple{AbstractMatrix{Float64},AbstractVector{Float64},SparseMatrixCSC{Float64,Int64},AbstractMatrix{Float64}}, λ::Float64)
    A,y,C,PCt = p
    u0 = mean(u, dims=2)
    mul!(PCt, cov(u'), C')
    mul!(A, -0.5PCt, (λ*C*PCt+I)\C);
    Threads.@threads for p in axes(u,2)
        du[:,p]  .= A*u[:, p] .+ (I+2λ*A)*((I+λ*A)*(PCt*y)+A*u0)
    end
end