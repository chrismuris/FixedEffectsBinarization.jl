export ConvertPanelToDiffCS_2

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