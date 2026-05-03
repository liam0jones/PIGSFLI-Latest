"""Data type to handle snapshots"""
mutable struct SnapshotHandler 
    path_list::Vector{String} 
    snapshot_frequency_list::Vector{Int64}
    handler_name_lookup::Dict{String,Int64}
    n_entries::Int64
    next_snapshot_list::Vector{Int64}
end
"""Initialize an empty snapshot handler"""
SnapshotHandler() = SnapshotHandler(Vector{String}(), Vector{Int64}(), Dict{String,Int64}(), 0, Vector{Int64}())
function add!(sh::SnapshotHandler, path::String, updata_every_steps::Int64, handler_name::String)
    if haskey(sh.handler_name_lookup, handler_name)
        error("Snashot handler names must be unique. Tried to add new snapshot handler with already existing name ", handler_name, ".")
    end
    sh.n_entries += 1
    push!(sh.path_list, path) 
    push!(sh.snapshot_frequency_list, updata_every_steps) 
    push!(sh.next_snapshot_list, updata_every_steps)
    sh.handler_name_lookup[handler_name] = sh.n_entries
    return nothing
end 
"""Write data to file. File is chosen by handler_name, use jld2 to save object to file."""
function Base.write(sh::SnapshotHandler, handler_name::String, data) 
    index = sh.handler_name_lookup[handler_name] 
    path = sh.path_list[index] 
    jldsave(path; data)
    return nothing
end  
"""Read data from file. File is chosen by handler_name, jld2 is used to load data."""
function Base.read(sh::SnapshotHandler, handler_name::String) 
    index = sh.handler_name_lookup[handler_name]
    path = sh.path_list[index]
    return load(path, "data")
end 
"""Checks if it is time to do a snapshot. Pass -1 to skip check and return true."""
function time_for_snapshot(sh::SnapshotHandler, handler_name::String, i::Int64)
    i == -1 && return true  

    index = sh.handler_name_lookup[handler_name]
    if sh.snapshot_frequency_list[index] == 0
        return false
    end
    # need to check like this as we can only save after moving to cpu before a measurement 
    # and steps in between measurements are variable
    do_snapshot = i >= sh.next_snapshot_list[index]
    if do_snapshot
        sh.next_snapshot_list[index] += sh.snapshot_frequency_list[index]
    end
    return do_snapshot
end