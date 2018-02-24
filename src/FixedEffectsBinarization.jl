module FixedEffectsBinarization

using StatsFuns.logistic
using DataFrames
using Optim: optimize, minimizer, LBFGS, NewtonTrustRegion
using Distributions
using Plots

include("strucs.jl")
include("datageneration.jl")
include("datahandling.jl")
include("estimation.jl")
include("postestimation.jl")

end