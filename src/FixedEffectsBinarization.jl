module FixedEffectsBinarization

using StatsFuns.logistic
using DataFrames: by, join, ModelMatrix, ModelFrame, model_response
export NewtonRaphson, score_FEBClogit_2!, Hessian_FEBClogit_2!, ConvertPanelToDiffCS_2, FixedEffectsBinaryChoiceLogit_2, score_FEOL_2!, Hessian_FEOL_2!, FixedEffectsOrderedLogit_2

function FixedEffectsOrderedLogit_2(formula,data,isymbol,tsymbol)
    
    # Convert the DataFrame + formula + (i,t)-indicators into vectors and matrices.
    y,X = ConvertPanelToDiffCS_2(formula,data,:i,:t)
    n, K = size(X)
    
    # Count how many additional columns there will be.
    cuts1 = sort(unique(y[:,1]))[2:end] #2:end: skip the first one
    cuts2 = sort(unique(y[:,2]))[2:end]
    n_1 = length(cuts1)-1 # normalization: first cut point in first period at 0.
    n_2 = length(cuts2)
    
    # Go for it!
    b_hat = NewtonRaphson(score_FEOL_2!,Hessian_FEOL_2!,zeros(K+n_1+n_2),y,X)
    
    return b_hat
end

function score_FEOL_2!(J::AbstractArray,b::AbstractArray,y,X)
    
    # Extract the length
    n, K = size(X)
    
    # Initialize the Jacobian to zero.
    J .= 0
    
    # Compute what the cut points are, from observing the dependent variable.
    cuts1 = sort(unique(y[:,1]))[2:end] #2:end: skip the first one
    cuts2 = sort(unique(y[:,2]))[2:end]
    n_1 = length(cuts1)-1 # normalization: first cut point in first period at 0.
    n_2 = length(cuts2)
    
    # Set up the matrix of dummies.
    Z = zeros(n,n_1+n_2)
    
    
    for i in 1:length(cuts1)
        for j in 1:length(cuts2)
                
            d1 = y[:,1].>=cuts1[i]
            d2 = y[:,2].>=cuts2[j]
            dbar = (d1 + d2) .== 1
            
            if i > 1
                Z[:,i-1] = 1
            end
            Z[:,n_1+j] = -1
            
            J = J + score_FEBClogit_2!(J,b,[d1 d2 dbar],[X Z])
            
            Z = zeros(n,n_1+n_2)
                
        end
    end
    
    return J
    
end

function Hessian_FEOL_2!(H::AbstractArray,b::AbstractArray,y,X)
    n,K = size(X) # Make sure a constant is added to X when this is called.
    #length(b) == K ? nothing : error("Length of b does not match that of X plus a constant.")
    H .= 0
    
    cuts1 = sort(unique(y[:,1]))[2:end]
    cuts2 = sort(unique(y[:,2]))[2:end]
    n_1 = length(cuts1)-1
    n_2 = length(cuts2)
    
    # Set up the matrix of dummies.
    Z = zeros(n,n_1+n_2)
    
    
    for i in 1:length(cuts1)
        for j in 1:length(cuts2)
                
            d1 = y[:,1].>=cuts1[i]
            d2 = y[:,2].>=cuts2[j]
            dbar = (d1 + d2) .== 1
            
            if i > 1
                Z[:,i-1] = 1
            end
            Z[:,n_1+j] = -1
            
            H = H + Hessian_FEBClogit_2!(H,b,[d1 d2 dbar],[X Z])
            
            Z = zeros(n,n_1+n_2)
                
        end
    end
    
    return H
end


"""
    FixedEffectsBinaryChoiceLogit_2(formula,data,isymbol,tsymbol; constant = false)

Implements Andersen's/Chamberlain's fixed effects estimator for a binary choice model with logistic errors.
This implementation is restricted to 2 time periods.

It takes a formula object, e.g. @formula(y~X1+X2), a data set, and symbols
that indicate which columns in the data set are the cross-section and time
indicators.

It prepares the data and then passes it to NewtonRaphson, calling the score
and Hessian functions specific to binary choice.
"""
function FixedEffectsBinaryChoiceLogit_2(formula,data,isymbol,tsymbol; constant = false)
    
    # Convert the DataFrame + formula + (i,t)-indicators into vectors and matrices.
    y,X = ConvertPanelToDiffCS_2(formula,data,:i,:t)
    
    # Add a constant term if required.
    n, K = size(X)
    if constant
        X = [ones(n) X]
    end
    
    # Prep Y-matrix for optimization routine.
    Y = [y[:,1] y[:,2] (y[:,1]+y[:,2].==1)]
    
    b_hat = NewtonRaphson(score_FEBClogit_2!,Hessian_FEBClogit_2!,zeros(K),Y,X)
    
    return b_hat
end


"""
    ConvertPanelToDiffCS_2(formula,data,idsymbol,tsymbol)

Takes a panel data set in long form (one row represents measurements
on a cross-section unit at a given time) and converts it into a
cross-sectional data set with one row representing differences.

It outputs two matrices:

`Y`, an `n` by 2 matrix, where the first column is the depdent variables
   in the first period, and `y2` is the dependent variables in the second.
`X`, an `n` by `K` matrix, with first differences in all explanatory 
   variables.
"""
function ConvertPanelToDiffCS_2(formula,data,idsymbol,tsymbol)
    
    # Takes a data frame with panel data in two time periods, and returns the data
    #   in a form that is ready for this package's optimization procedure.
    
    # First, sort the data
    sort!(data, cols = [idsymbol, tsymbol])
    
    # Second, remove the incomplete records.
    dfcount = by(data,idsymbol, df -> length(df[tsymbol]))
    complete_id = dfcount[dfcount[:x1].==2,:]
    data = join(data,complete_id,on = idsymbol, kind = :inner)
        
    # Third, extract the model matrix ("X"s)
    MF = ModelFrame(formula,data)
    X = ModelMatrix(MF).m[:,2:end]
    
    # Take differences, exploiting the fact that T=2 and the data was previously balanced.
    n = size(X)[1] #new effective size after dropping the incomplete ones
    dX = X[filter(iseven,1:n),:] - X[filter(isodd,1:n),:]
    
    # Fourth, extract the response variable.
    y = model_response(MF)
    y1 = y[filter(isodd,1:n)]
    y2 = y[filter(iseven,1:n)]
    ybar = 1.0*((y1 + y2) .== 1)
    
    return [y1 y2], dX
    
end

"""
    score_FEBClogit_2!(J,b,y,X)

Computes the score for a two-period fixed effects binary choice model.

Its inputs are `J`, a Kx1 Vector that will be assigned the score,
`b` a value of the regression parameter,
`y` an `n` by `3` matrix consisting of `y1`, `y2`, and `y1+y2 == 1`.
`X` an `n` by `K` matrix with differenced regressors.
"""
function score_FEBClogit_2!(J::AbstractArray,b::AbstractArray,y,X)
    K = size(X)[2] # Make sure a constant is added to X when this is called.
    length(b) == K ? nothing : error("Length of b does not match that of X plus a constant.")
    Ztheta = X*b
    J = [mean( (y[:,3]).*(( y[:,2] .- logistic.(Ztheta)) .* X)[:,i]) for i in 1:K]
end

"""
    Hessian_FEBClogit_2!(H,b,y,X)

Computes the score for a two-period fixed effects binary choice model.

Its inputs are `H`, a KxK matrix that will be assigned the Hessian,
`b` a value of the regression parameter,
`y` an `n` by `3` matrix consisting of `y1`, `y2`, and `y1+y2 == 1`.
`X` an `n` by `K` matrix with differenced regressors.
"""
function Hessian_FEBClogit_2!(H::AbstractArray,b::AbstractArray,y,X)
    n,K = size(X) # Make sure a constant is added to X when this is called.
    length(b) == K ? nothing : error("Length of b does not match that of X plus a constant.")
    Ztheta = X*b
    H = -1/n*((y[:,3]).*(logistic.(Ztheta)).*(1.-logistic.(Ztheta)) .* X)'*X
end

"""
    NewtonRaphson(objective!, gradient!, b0, y, X)

Compute the maximum likelihood estimate given a score `objective!` and Hessian function `gradient!`.
The parameter `b0` is the starting value, which must be guessed by the user.
Measurements on the dependent variable `y` are in a Matrix of height `n`.
    It is a matrix because it may contain multiple output variables, as in the panel data case we are interested in here.
Measurements on the `K` explanatory variables (include a constant!) come in the `n` by `K` array `X`.

`NewtonRaphson` passes `y` and `X` on to the score and Hessian, 
which must be of the form

    objective!(J,b,y,X)
    gradient!(H,b,y,X)

where they will overwrite their first argument with the computed values, 
for a given parameter value b, and the data y, X that are passed to them.

Optional arguments, with defaults:

    maxiter = 1000
    abstol = 1e-10
    relax = 0.9

are, in order, the maximum number of iterations, 
a stopping criterion for both the estimate and the score,
and a parameter that controls how relaxed the NR algorithm is. 
"""
function NewtonRaphson(objective!, gradient!, b0, y, X; maxiter = 1000, abstol = 1e-10, relax = 0.9)
    
    # objective! is a function to set to zero (score-type) with first aargument being
    #      a replaceable array of length K, and the second being a value for the
    #      parameter we are trying to estimate.
    # gradient! is the gradient of that function, with first argument a replaceable
    #      KxK matrix, and second the parameter value.
    
    K = length(b0)
    
    f0 = objective!(zeros(K),b0,y,X)
    J0 = gradient!(zeros(K,K),b0,y,X)
    xn = b0
    fn = f0
    Jn = J0
    
    for i in 1:maxiter

        xnp1 = xn - relax*pinv(Jn)*fn
        fn = objective!(fn,xn,y,X)
        Jn = gradient!(Jn,xn,y,X)
    
        discrep = ((xnp1-xn)'*(xnp1-xn)) / length(xnp1) 
        if discrep < abstol && abs(findmax(fn)[1])<abstol
            break
        end
    
        xn = xnp1
        
        if i == maxiter
            println("Did not converge after $maxiter iterations.")
        end
    end
    
    return xn
end    

end # module
