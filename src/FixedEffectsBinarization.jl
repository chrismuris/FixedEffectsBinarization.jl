module FixedEffectsBinarization

using StatsFuns.logistic
using DataFrames
using Optim: optimize, minimizer, LBFGS, NewtonTrustRegion
using Distributions

export FELT

include("datageneration.jl")
include("datahandling.jl")
include("estimation.jl")

# Each application is characterized by:
# 1. (y,X) (well, formula, data, :i, : for the user)
# 2. Which cut points to choose
# 3. How to interpolate the estimated function in between the cut points.
# If the outcome variable is discrete, then generally 2. will correspond to all points in the support of the dependent variable, and 3. will be a step function.
# If the outcome variable is continuous, then generally 2. will consist of some cleverly chosen points, and 3. will be a linear interpolation, or a B-spline, or something like that. I like linear interpolation because it is consistent with censoring.

mutable struct FELT
    formula
    data::DataFrame
    isymbol::Symbol
    tsymbol::Symbol
    y::Array{Float64,2}
    X::Array{Float64,2}
    y1s::Vector{Float64}
    y2s::Vector{Float64}
    b_hat::Vector{Float64}
    gamma_1_hat::Vector{Float64}
    gamma_2_hat::Vector{Float64}
    discrete::Bool
end

function FELT(formula,data,isymbol,tsymbol; discrete = true, levels = 5)
    
    # This only works for discrete!
    
    y, X = ConvertPanelToDiffCS_2(formula,data,isymbol,tsymbol)
    n,K = size(X)
    
    if discrete 
        cuts1 = sort(unique(y[:,1]))[2:end] #2:end: skip the first one
        cuts2 = sort(unique(y[:,2]))[2:end]
        
        print(cuts1)
        print(cuts2)
        
        #error("Stop")

    else  #well then it must be continuous.
       
        # Standard number of levels is 5???
        q_points = collect(0:(1/(levels+1)):1)[2:(end-1)]
        
        cuts1 = quantile(y[:,1], q_points)[:,1]
        cuts2 = quantile(y[:,2], q_points)[:,1]

    end

    # Need these for extracting b, gamma1, gamma2
    n_1 = length(cuts1) - 1
    n_2 = length(cuts2)
    
    theta_hat = FixedEffectsOrderedLogit_2(y,X,cuts1,cuts2)
    b_hat = theta_hat[1:K]
    gamma_1_hat = [0;theta_hat[(K+1):(K+n_1)]]
    gamma_2_hat = theta_hat[(K+n_1+1):(K+n_1+n_2)]
    
    return FELT(formula,data,isymbol,tsymbol,
                y,X,cuts1,cuts2,
                b_hat,gamma_1_hat,gamma_2_hat,
                discrete)

end



end