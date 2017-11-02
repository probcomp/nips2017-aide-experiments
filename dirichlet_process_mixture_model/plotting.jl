function get_xlim(data::DataFrame)
    num_particles_column = convert(Vector{Int},data[COL_NUM_PARTICLES])
    
    # add some padding on the left and right (in logspace)
    #return [10/1.1, 100*1.1] # TODO
    return [minimum(num_particles_column)/1.1, maximum(num_particles_column)*1.1]
end

function add_filled_circle(x, y, color)
    #ax = plt[:gca]()
    #xlim = ax[:get_xlim]()
    #ylim = ax[:get_ylim]()
    plt[:scatter]([x], [y], color=color, s=50, edgecolor="black", zorder=100)
    #ax[:set_xlim](xlim)
    #ax[:set_ylim](ylim)
end

function draw_arrow(x, y, xlen, ylen)
    ax = plt[:gca]()
    ax[:arrow](x, y, xlen, ylen, fc="black", ec="black",
        transform=ax[:transAxes], length_includes_head=true,
        head_width=0.05, head_length=0.05)
end

function text(x, y, content, fontsize)
    ax = plt[:gca]()
    plt[:text](x, y, content, fontsize=fontsize, transform=ax[:transAxes])
end


