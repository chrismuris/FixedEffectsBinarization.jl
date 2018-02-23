module FixedEffectsBinarization

using StatsFuns.logistic
using DataFrames: by, join, ModelMatrix, ModelFrame, model_response, sort!
using Optim: optimize, minimizer, LBFGS, NewtonTrustRegion

include("datageneration.jl")
include("datahandling.jl")
include("estimation.jl")

end