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

# Each application is characterized by:
# 1. (y,X) (well, formula, data, :i, : for the user)
# 2. Which cut points to choose
# 3. How to interpolate the estimated function in between the cut points.
# If the outcome variable is discrete, then generally 2. will correspond to all points in the support of the dependent variable, and 3. will be a step function.
# If the outcome variable is continuous, then generally 2. will consist of some cleverly chosen points, and 3. will be a linear interpolation, or a B-spline, or something like that. I like linear interpolation because it is consistent with censoring.



end