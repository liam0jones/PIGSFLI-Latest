abstract type AbstractVectorData end

@inline Base.size(a::AbstractVectorData) = (a.data.size,)
@inline Base.IndexStyle(::Type{<:AbstractVectorData}) = IndexLinear() 
@inline Base.similar(a::AbstractVectorData) = typeof(a)(similar(a.data,eltype(a.data),length(a.data)))
@inline Base.similar(a::AbstractVectorData, dims::Int64) = typeof(a)(similar(a.data,eltype(a.data),dims))
@inline Base.getindex(a::AbstractVectorData, i::Int) = a.data[i]
@inline Base.getindex(a::AbstractVectorData, i::Any) = a.data[i]
@inline Base.setindex!(a::AbstractVectorData, v, i::Int) = (a.data[i] = v) 
@inline Base.setindex!(a::AbstractVectorData, v, i::Any) = (a.data[i] = v) 
@inline Base.firstindex(a::AbstractVectorData) = firstindex(a.data) 

abstract type AbstractMatrixData end

@inline Base.size(a::AbstractMatrixData) = (a.data.size,)
@inline Base.IndexStyle(::Type{<:AbstractMatrixData}) = IndexLinear() 
@inline Base.similar(a::AbstractMatrixData) = typeof(a)(similar(a.data,eltype(a.data),length(a.data)))
@inline Base.similar(a::AbstractMatrixData, dims::Int64) = typeof(a)(similar(a.data,eltype(a.data),dims))
@inline Base.getindex(a::AbstractMatrixData, i::Int) = a.data[i]
@inline Base.getindex(a::AbstractMatrixData, i::Int, j::Int) = a.data[i,j]
@inline Base.getindex(a::AbstractMatrixData, i::Any, j::Any) = a.data[i,j]
@inline Base.setindex!(a::AbstractMatrixData, v, i::Int) = (a.data[i] = v) 
@inline Base.setindex!(a::AbstractMatrixData, v, i::Int, j::Int) = (a.data[i,j] = v) 
@inline Base.setindex!(a::AbstractMatrixData, v, i::Any, j::Any) = (a.data[i,j] = v) 
@inline Base.firstindex(a::AbstractMatrixData) = firstindex(a.data)
@inline Base.axes(a::AbstractMatrixData, i::Int) = axes(a.data, i)

abstract type TrialState end

abstract type ModelSystem end