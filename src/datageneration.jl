
export ConvertInterval, DGP_FEOL_2

function ConvertInterval(ystar,thresholds) 
    # This function translates a vector of latent variables to a vector of interval-censored outcomes.
     
    y = similar(ystar)
    for i in 1:length(ystar)
        y[i] = sum(ystar[i].>thresholds) + 1
    end
    return y
end
#df[:y] = ConvertInterval(df[:ystar],thresholds)

function DGP_FEOL_2(n, beta0k,thresholds1,thresholds2)

    T = 2 #hard-code T=2?
    K = 2 # also hard-coded
    nT = n*T
    
    beta0 = beta0k + zeros(K,1)
    D_X = Normal(0,1)
    D_u = Logistic()

    # Generate data frame.
    df = DataFrame(
        i = repeat(1:n, inner = T),
        t = repeat(1:T, outer = n),
        X1 = rand(D_X,n*T),
        X2 = rand(D_X,n*T),
        u = rand(D_u,n*T)
    );

    df_a = by(df, [:i], y-> 0.2*(mean(y[:X1]) + mean(y[:X2])))
    df_a[:a] = df_a[:x1]
    delete!(df_a,:x1)
    df = join(df,df_a,on = :i)
    df[:ystar] = 0 + df[:X1]*beta0[1] + df[:X2]*beta0[2] - df[:u] + df[:a]
    
    # This only works because the data is perfectly shaped and T=2.
    df[:y] = 0
    df[filter(isodd,1:nT),:y] = ConvertInterval(df[filter(isodd,1:nT),:ystar],thresholds1)
    df[filter(iseven,1:nT),:y] = ConvertInterval(df[filter(iseven,1:nT),:ystar],thresholds2)
    
    return df
end