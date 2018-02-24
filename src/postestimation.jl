export plot_h, plot_dist, plot_dist_control

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