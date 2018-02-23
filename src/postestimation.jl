using Plots

export plot_h

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
    plot!(legend=:topleft)
    
end