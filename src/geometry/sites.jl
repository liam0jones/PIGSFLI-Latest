struct Sites <: AbstractVectorData 
    data::Vector{Int64} 
    geometry::String 
    l_max::Int64
    L::Int64
    D::Int64
    M::Int64
    len::Int64
end
function Sites(geometry::String, l_max::Int64, L::Int64, D::Int64, M::Int64) 
    n_sites::Int64 = 0
    # total number of sites
    if (D==1 || L==2)
        n_sites = l_max
    elseif (D==2)
        if (geometry=="square")
            n_sites = l_max^D
        elseif (geometry=="strip")
            n_sites = l_max*L
        else 
            error(@sprintf "Geometry '%s' not implemented, use 'square' or 'strip'." geometry)
        end
    else 
        error(@sprintf "%d dimensions not implemented, use D=1 or D=2." D)
    end

    return Sites(zeros(Int64,n_sites), geometry, l_max, L, D, M, n_sites)
end

function create_sub_sites(l_max::Int64, L::Int64, D::Int64, M::Int64,geometry::String)::Sites
    # allocate sites
    sites = Sites(geometry, l_max, L, D, M)
    # fill sites
    if (D==1 || L==2)
        init_1D!(sites,l_max)
    elseif (D==2)
        if (geometry=="square")
            init_2D_square!(sites, l_max, L)

        elseif (geometry=="strip")
            init_2D_strip!(sites,L)

        end
    end
    return sites
end

function init_1D!(sites::Sites, l_max::Int64)
    for site in 0:l_max-1
        sites[site+begin] = site
    end
end

function init_2D_square!(sites::Sites, l_max::Int64, L::Int64)  
    ctr=0 
    y=0 
    _curr_idx::Int64 = 0
    for l=0:l_max-1
        next_sub_site = l 
        sites[_curr_idx+begin] = next_sub_site
        _curr_idx += 1 
        
        for j=1:l
            next_sub_site = l+j*L 
            sites[_curr_idx+begin] = next_sub_site
            _curr_idx += 1
            if (j==l) 
                for i=1:l 
                    next_sub_site -= 1 
                    sites[_curr_idx+begin] = next_sub_site
                    _curr_idx += 1
                end
            end
        end
        y += L 
        ctr += 1 
    end
end

function init_2D_strip!(sites::Sites, L::Int64) 
    _curr_idx::Int64 = 0
    m_max = sites.len 
    next_sub_site = -1
    horizontal_direction = +1
    horizontal_direction_old = +1
    vertical_direction = 0
    ctr = 0

    while _curr_idx <= m_max
        if ctr == L 
            vertical_direction = +L
            horizontal_direction = 0
            ctr=0
        elseif _curr_idx > 3 && ctr == 1
            vertical_direction = 0
            horizontal_direction = (-1)*horizontal_direction_old
            horizontal_direction_old = horizontal_direction 
        end
        next_sub_site += (horizontal_direction+vertical_direction)
        sites[_curr_idx+begin] = next_sub_site 
        _curr_idx += 1
        ctr += 1
    end

end