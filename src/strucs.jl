export FELT, newFELT, NLDID, newNLDID

# Each application is characterized by:
# 1. (y,X) (well, formula, data, :i, : for the user)
# 2. Which cut points to use
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

# Building on FELT, a nonlinear difference-in-differences structure 
#   consists of:
#     1. a FELT object, containing control outcomes and estimation results
#     2. treatment indicator
#     3. treatment data
#     4. counterfactual outcomes.
#     5. Linear DiD estimate

mutable struct NLDID
    treatsymbol::Symbol
    felt::FELT
    t::Vector{Float64}
    y_treat::Vector{Float64}
    X_treat::Array{Float64,2}
    linearATT
    nonlinearATT
end

function newFELT(formula,data,isymbol,tsymbol; ys = false, discrete = true, levels = 10)
    
    # This only works for discrete!
    
    y, X = ConvertPanelToDiffCS_2(formula,data,isymbol,tsymbol)
    n,K = size(X)
    
    if ys !== false
        
        cuts1 = ys
        cuts2 = ys
        
    else
        
        if discrete 
            cuts1 = sort(unique(y[:,1]))[2:end] #2:end: skip the first one
            cuts2 = sort(unique(y[:,2]))[2:end]
            
        else  #well then it must be continuous.
            
            # Compute `levels` equi-spaced quantiles.
            q_points = collect(0:(1/(levels+1)):1)[2:(end-1)]
            
            y1low = findmin(y[:,1])[1]
            y2low = findmin(y[:,2])[1]
            
            cuts1 = setdiff(quantile(y[:,1], q_points)[:,1],y1low)
            cuts2 = setdiff(quantile(y[:,2], q_points)[:,1],y2low)
            
        end
        
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

function newNLDID(formula,data,isymbol,tsymbol,treatsymbol; ys = false, discrete = true, levels = 10)
    
    # First split
    data_0 = data[data[treatsymbol].==0,:]
    data_1 = data[data[treatsymbol].==1,:]
    
    # Create the FELT instance
    f = newFELT(formula,data_0,isymbol,tsymbol; ys = ys, discrete = discrete, levels = levels)
    
    # Extract the treat data.
    t = Vector(data_1[tsymbol])
    MF = ModelFrame(formula,data_1)
    X_treat = ModelMatrix(MF).m[:,2:end]
    y_treat = model_response(MF)
    
    y0 = y_treat[t .== 0]
    y1 = y_treat[t .== 1]
    X0 = X_treat[t .== 0,:]
    X1 = X_treat[t .== 1,:]
    
    if discrete
        yLO = minimum(y1)
        yHI = maximum(y1)
        y1_cf = counterfactual_discrete(y0,yLO,yHI,
                                        f.gamma_1_hat,f.y1s,
                                        f.gamma_2_hat,f.y2s)
        nlATT = [mean(y1)-y1_cf[2], mean(y1)-y1_cf[1]]
    else
        
        y1_cf = counterfactual_continuous(y0,X0,X1,f.b_hat,f.y1s,f.gamma_1_hat,f.y2s,f.gamma_2_hat,minimum(y_treat),maximum(y_treat))
        nlATT = mean(y1) - mean(y1_cf)
    end
    
    return NLDID(treatsymbol,f,t,y_treat,X_treat,"TBD",nlATT)
end
