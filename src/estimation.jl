export fgh_FEOL_2, FixedEffectsOrderedLogit_2

"""
    FixedEffectsOrderedLogit_2(formula,data,isymbol,tsymbol; b0, solver)

computes the composite conditional MLE for the FEOL model with two time periods.

The formula, data, isymbol, tsymbol tell the algorithm which model and data to use.
If you want to supply a starting value, assign `b0` (or it will start from zeros).
The default solver is `LBFGS` (using `Optim.jl`). Set `solver=Newton()` to use NewtonRaphson
(analytical Hessian will be used), or set it to...

- `NewtonTrustRegion`
- `AcceleratedGradientDescent`
- `NelderMead`
- ... (see the documentation for `Optim.jl`),

but in that case you must issue `using Optim` where you are issuing the estimation command.
"""
function FixedEffectsOrderedLogit_2(y,X,cuts1,cuts2;b0=false,solver = LBFGS(), verbose = true)
    
    # set starting value.
    if b0 == false
        n,K = size(X)
        # Count how many additional columns there will be.
        n_1 = length(cuts1) - 1
        n_2 = length(cuts2)
        b_start = zeros(K+n_1+n_2)
        
        println()
    else
        b_start = b0
    end
    
    # Define the objective, gradient, and Hessian
    objective, gradient!, Hessian! = fgh_FEOL_2(b_start,y,X,cuts1,cuts2)
    
    # Perform the optimization
    println("========================================")
    println("Starting the first minimization attempt.")
    println("========================================")
    opt_res = optimize(objective, gradient!, Hessian!, b_start, solver)

    # Report optimization diagnostics to user.
    if verbose
        print(opt_res)
    end

    if opt_res.g_converged
        println("")
        println("===============================")
        println("Converged on the first attempt.")
        println("===============================")
        println("")
    else
        opt_res_1 = opt_res
        println("")
        println("")
        println("====================================================")
        println("Starting a second round, using Newton Trust Region..")
        println("====================================================")
        println("")
        opt_res = optimize(objective, gradient!, b_start, NewtonTrustRegion())
        if verbose
            print(opt_res)
        end
        println("")
        println("")
        println("======================================================")
        println("Further decrease of $(opt_res_1.minimum - opt_res.minimum).")
        println("======================================================")
    end

    return opt_res.minimizer
end

"""
    fgh_FEOL_2(theta,y,X)

computes the (negative of the) log-likelihood, score, and 
Hessian for the fixed effects ordered logit model.
"""
function fgh_FEOL_2(theta::Array,y::Array,X::Array, cuts1, cuts2)
    
    # The data (y,X) comes in cross-sectionalized: n high, 2 cq K wide.
    # Settle some dimensions first.
    n, K = size(X) 
    n_1 = length(cuts1) - 1 
    n_2 = length(cuts2)
    
    #
    # Probably should do some checks on the number of observations for each set of cut points.
    #
    
    # theta is a vector of coefficients. First K are beta. 
    function min_ll(theta::Array{Float64})
        
#         println(cuts1)
#         println(cuts2)
#         println(n_1)
#         println(n_2)
#         println(K)
        
        # Initialize
        Xb = X*theta[1:K]
        ll = 0.
        
        for i in 1:(n_1+1)
            
            d1 = y[:,1].>=cuts1[i]
            
            for j in 1:n_2
                
                d2 = y[:,2].>=cuts2[j]
                dbar = (d1.+d2).==1
                
                if i == 1
                    h = logistic.(Xb .- theta[K+n_1+j])
                else
                    h = logistic.(Xb .+ theta[K+i-1] .- theta[K+n_1+j])
                end
                    
                ll = ll - mean(dbar.*(d2.*log.(h) .+ (1.-d2).*log.(1.-h)))
                
            end
        end
        
        return ll
        
    end
    
    function min_score!(J::Array,theta::Array)

       # Initialize the Jacobian to zero.
        J[:] .= 0.
        # Set the linear index.
        Xb = X*theta[1:K] 

        for i in 1:length(cuts1)

            d1 = y[:,1].>=cuts1[i]

            for j in 1:length(cuts2)

                Jnew = zeros(K+n_1+n_2) 

                d2 = y[:,2].>=cuts2[j]
                dbar = (d1 .+ d2) .== 1

                if i == 1
                    scalar_in = dbar.*(d2 .- logistic.(Xb .- theta[K+n_1+j]))
                else 
                    scalar_in = dbar.*(d2 .- logistic.(Xb .+ theta[K+i-1] .- theta[K+n_1+j]))
                end

                Jnew[1:K] = X'*scalar_in/n

                cut_H = mean(scalar_in)
                if i>1
                    Jnew[K+i-1] = cut_H
                end
                Jnew[K+n_1+j] = -cut_H

                # Assemble it.
                J[:] = J - Jnew
            end
        end
    end
    
    function min_Hess!(H::Array,theta::AbstractArray)
        
        # Initialize the Jacobian to zero.
        H[:] .= 0.

        # Predefine the linear index.
        Xb = X*theta[1:K] 

        for i in 1:length(cuts1)

            d1 = y[:,1].>=cuts1[i]

            for j in 1:length(cuts2)

                # Initialize the Hessian for this case
                Hnew = zeros(K+n_1+n_2,K+n_1+n_2) 

                d2 = y[:,2].>=cuts2[j]
                dbar = (d1 + d2) .== 1

                # Compute the top-left block.
                if i == 1
                    Lambda = dbar.*(logistic.(Xb - theta[K+n_1+j]))
                else 
                    Lambda = dbar.*(logistic.(Xb + theta[K+i-1] - theta[K+n_1+j]))
                end
                Hnew[1:K,1:K] = -(1/n)*(Lambda.*(1-Lambda) .* X)'*X

                # Compute the bottom-right block.
                cut_H = -mean(Lambda.*(1-Lambda))
                if i>1
                    Hnew[K+i-1,K+i-1] = cut_H
                end
                Hnew[K+n_1+j,K+n_1+j] = cut_H

                # Compute the off-diagonal blocks.        
                H_off = [ -mean( Lambda.*(1-Lambda) .* X[:,k]) for k in 1:K ]

                if i>1
                    Hnew[1:K,K+i-1] = H_off
                    Hnew[K+i-1,1:K] = H_off'
                end
                Hnew[1:K,K+n_1+j] = -H_off
                Hnew[K+n_1+j,1:K] = -H_off'

                H[:] = H - Hnew

            end
        end
    end
    
    return min_ll, min_score!, min_Hess!
end