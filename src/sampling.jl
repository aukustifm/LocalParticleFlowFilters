# Libraries ____________________
using LinearAlgebra, Statistics, Distributions
# ______________________________

# Functions ____________________
"""
    Inv4Resampling(w,u) 

Obtain indices \$A = (A_1,\\dots,A_n)\$ for resampling through inverse transform sampling over the importance weights \$w = (w_1,\\dots,w_n)\$  and uniform samples \$u = (u_1,\\dots,u_n)\$ ordered as \$(u_1 \\leq u_2 \\leq \\cdots \\leq u_n)\$.
"""
function Inv4Resampling(w::Vector{Float64}, u::StepRangeLen)
	N = size(w,1); 
    A = Array{Int64,1}(undef,N);
    
    m::Int64 = 1; s::Float64 = w[1];
    for n = 1:Int(N)
        while s < u[n] && m <= N-1
            m += 1;
            s += w[m];
        end
        A[n] = m;
    end
    return A
end;

"""
    SystematicResampling(w) 
    
Returns the resampling \$A = (A_1,\\dots,A_n)\$ from importance weights \$w = (w_1,\\dots,w_n)\$ using Systematic Resampling.
"""
function SystematicResampling(w::Vector{Float64})
    N = size(w,1);
    u = (0:N-1)./N .+ rand()/N
    return Inv4Resampling(w, u)
end;

"""
    StratifiedResampling(w) 
    
Returns the resampling \$A = (A_1,\\dots,A_n)\$ from importance weights \$w = (w_1,\\dots,w_n)\$ using Stratified Resampling.
"""
function StratifiedResampling(w::Vector{Float64})
    N = size(w,1);
    u = ((0:N-1) .+ rand(N))./N
    return Inv4Resampling( w, u )
end;

"""
    MultinomialResampling(w) 
    
Returns the resampling \$A = (A_1,\\dots,A_n)\$ from importance weights \$w = (w_1,\\dots,w_n)\$ using Multinomial Resampling.
"""
function MultinomialResampling(w::Vector{Float64})
    N = size(w,1);
    t = cumsum(-log.(rand(N+1)), dims=1)
    return Inv4Resampling( w, t./t[end] )
end;
