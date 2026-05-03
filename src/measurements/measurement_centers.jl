function get_measurement_centers(beta::Float64) 
    measurement_centers = Vector{Float64}()
    num_centers=25  # use odd number please.
    tau_center = beta/(2*num_centers) 
    for i = 1:num_centers
        append!(measurement_centers,tau_center)
        tau_center+=(beta/(num_centers)) 
    end
    return measurement_centers 
end