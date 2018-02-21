module FixedEffectsBinarization

using StatsFuns.logistic

export NewtonRaphson, score_FEBClogit_2!, Hessian_FEBClogit_2!

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
    
    K = size(X)[2]
    Xb = X*b0
    
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
