export plot_h, plot_dist, plot_dist_control, counterfactual_discrete, plot_dist_treat

function plot_h(f::FELT)
    
    x1 = f.gamma_1_hat
    y1 = f.y1s
    x2 = f.gamma_2_hat
    y2 = f.y2s
    
    if f.discrete
        plot(x1,y1-0.1, linetype = :steppost, linewidth = 5)
        plot!(x2,y2+0.1, linetype = :steppost, linewidth = 5)
    else
        plot(x1,y1-0.1, linewidth = 5)
        plot!(x2,y2+0.1, linewidth = 5)
    end
    
    xmin = min(findmin(x1)[1],findmin(x2)[1])
    xmax = max(findmax(x1)[1],findmax(x2)[1])
    extend = 0.1*(xmax-xmin)
    plot!(xlims=(xmin-extend,xmax+extend))
    
    yvals = sort(union(y1,y2))
    ymin = findmin(yvals)[1]
    ymax = findmax(yvals)[1]
    #yticks!(yvals)
    ylims!(ymin-1,ymax+1)
    p = plot!(legend=:topleft)

    return(p)
    
end

function plot_dist(f::FELT)
    return histogram2d(f.y[:,1],f.y[:,2], bins = f.y1s)
end

function plot_h(f::NLDID)
    plot_h(f.felt)
    # Just redirect to the plot function for a FELT object.
end

function plot_dist_control(f::NLDID)
    # Delegate to the function for FELT.
    plot_dist(f.felt)
end

function plot_dist_treat(f::NLDID)
    y1 = f.y_treat[f.t.==0]
    y2 = f.y_treat[f.t.==1]
    histogram2d(y1,y2, bins = f.felt.y1s)
end

############# 
# Get the treatment effect.
#############

function counterfactual_discrete(Y0,YLOW,YHIGH,gamma_1,y1,gamma_2,y2)
   
    # Y0 is a vector of observed values in period 1 for which we want to compute
    #       counterfactuals in time 2.
    #
    # Should be passed in as 
    #     Y0 = f.y_treat[f.t .== 0] 
    # where f is a NLDID object.
    
    # YLOW is the lowest value the variable can take
    # YHIGH is the highest value the variable can take
    
    # y1 is the set of cut points used in period 1. These will exclude YLOW.
    # gamma_1 is the estimated thresholds on the latent variable, e.g. y*>=gamma_1[2]
    #    means y>=y1[2].
    # 
    # same for y2, gamma_2.
    #
    # 
    # Start by computing bounds on the latent variable.
      
    # lb, ub represent the lower and upper bound on the latent variable
    #   for period 1. They are vectors of the same size as Y0.
    lb = similar(Y0)*0.
    ub = similar(Y0)*0.
    n_treat = length(Y0)
    
    for i in 1:n_treat
        
        index = sum((Y0[i] .>= y1))

        if index == 0 
            lb[i] = -Inf
        else
            lb[i] = gamma_1[index]
        end

        if index == length(y1)
            ub[i] = Inf
        else
            ub[i] = gamma_1[index+1]
        end
    end
    
    # Now convert them into bounds on the outcome variable.
    
    # First the lower bounds. 
    lb_counterfactual = similar(Y0)*0.
    ub_counterfactual = similar(Y0)*0.
    
    for i in 1:n_treat
        
        index = sum(lb[i] .>= gamma_2)
        if index == 0
            lb_counterfactual[i] = YLOW
        else
            lb_counterfactual[i] = y2[index]
        end
        
        index = sum(ub[i] .>= gamma_2)
        if index == 0
            ub_counterfactual[i] = YLOW
        else
            ub_counterfactual[i] = y2[index]
        end
    end
      
    return mean(ub_counterfactual), mean(lb_counterfactual)
    
end