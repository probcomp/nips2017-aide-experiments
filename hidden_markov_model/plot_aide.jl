using DataFrames
using Query
using PyPlot
using LaTeXStrings

include("schema.jl")

# number of metainference runs to a solid or dashed linestyle
const linestyles = Dict([1 => "-", 100 => "--"])

# proposal name to color
const colors = Dict([PRIOR_PROPOSAL_NAME => "darkcyan", OPTIMAL_PROPOSAL_NAME => "red"])

# short names for the legend for the different types of proposals
proposal_short_names = Dict([PRIOR_PROPOSAL_NAME => "prior proposal", OPTIMAL_PROPOSAL_NAME => "optimal proposal"])

# where to read data from
const DATA_DIR = "data"

# where to write plots to
const PLOT_DIR = "plots"
if !Base.Filesystem.isdir(PLOT_DIR)
    Base.Filesystem.mkdir(PLOT_DIR)
end

const PLOT_FIGURE_SIZE = (3, 2.5)
const LEGEND_FIGURE_SIZE = (6.8, 1.4)

function plot_aide_estimates(data::DataFrame, gold_standard_name::String, proposal_name::String,
                             color::String)

    # for each meta-inference value, plot kl estimate vs number of particles
    num_metainference_list = sort(unique(data[COL_NUM_METAINFERENCE]))
    for num_metainference in num_metainference_list
        result = @from i in data begin
            @where ((i.gold_standard_name == gold_standard_name) &&
                    (i.proposal_name == proposal_name) &&
                    (i.num_metainference == num_metainference))
            @select {i.num_particles, i.aide_estimate}
            @collect DataFrame
        end
        plt[:plot](result[COL_NUM_PARTICLES], result[COL_AIDE_ESTIMATE],
                   linestyle=linestyles[num_metainference], color=colors[proposal_name])
    end
end

function plot_markers(data::DataFrame, gold_standard_name::String)
    for (num_particles, proposal_name) in [
        (1, PRIOR_PROPOSAL_NAME),
        (10, PRIOR_PROPOSAL_NAME),
        (100, OPTIMAL_PROPOSAL_NAME)]
        num_metainference = 1
        result = @from i in data begin
            @where ((i.gold_standard_name == gold_standard_name) &&
                    (i.proposal_name == proposal_name) &&
                    (i.num_metainference == num_metainference) &&
                    (i.num_particles == num_particles))
            @select {i.aide_estimate}
            @collect DataFrame
        end
        aide_estimate = result[COL_AIDE_ESTIMATE]
        @assert length(aide_estimate) == 1
        plt[:scatter]([num_particles], aide_estimate, color="white", s=75,
                      edgecolor="black", zorder=100)
    end

    # NOTE: labels are placed at hardcoded locations
    const fontsize = 26
    ax = plt[:gca]()
    plt[:text](0.05, 0.85, "A", fontsize=fontsize, transform=ax[:transAxes])
    plt[:text](0.5, 0.30, "B", fontsize=fontsize, transform=ax[:transAxes])
    plt[:text](0.87, 0.10, "C", fontsize=fontsize, transform=ax[:transAxes])
end


function draw_zero_line(ax, num_particles_list::Vector{Int})
    xlim = [minimum(num_particles_list), maximum(num_particles_list)]
    plt[:plot](xlim, [0, 0], color="black")
    ax[:set_xlim](xlim)
end

function format_axes(num_particles_list::Vector{Int})
    ax = plt[:gca]()
    draw_zero_line(ax, num_particles_list)
    ax[:set_xscale]("log")
    ax[:set_xlabel]("Number of particles")
    ax[:set_ylabel]("AIDE estimate (nats)")
    plt[:tight_layout](pad=0)
end

function plot_aide_estimate_exact_gold_standard(data::DataFrame, fname::String)
    num_particles_list = sort(unique(convert(Vector{Int}, data[COL_NUM_PARTICLES])))
    plt[:figure](figsize=PLOT_FIGURE_SIZE)
    plot_aide_estimates(data, EXACT_GOLD_STANDARD, PRIOR_PROPOSAL_NAME, colors[PRIOR_PROPOSAL_NAME])
    plot_aide_estimates(data, EXACT_GOLD_STANDARD, OPTIMAL_PROPOSAL_NAME, colors[OPTIMAL_PROPOSAL_NAME])
    plot_markers(data, EXACT_GOLD_STANDARD)
    format_axes(num_particles_list)
    plt[:savefig](fname)
end

function plot_aide_estimate_approximate_gold_standard(data::DataFrame, fname::String)
    num_particles_list = sort(unique(convert(Vector{Int}, data[COL_NUM_PARTICLES])))
    plt[:figure](figsize=PLOT_FIGURE_SIZE)
    plot_aide_estimates(data, APPROXIMATE_GOLD_STANDARD, PRIOR_PROPOSAL_NAME, colors[PRIOR_PROPOSAL_NAME])
    plot_aide_estimates(data, APPROXIMATE_GOLD_STANDARD, OPTIMAL_PROPOSAL_NAME, colors[OPTIMAL_PROPOSAL_NAME])
    plot_markers(data, APPROXIMATE_GOLD_STANDARD)
    format_axes(num_particles_list)
    plt[:savefig](fname)
end

function plot_legend(data::DataFrame, fname::String)
    plt[:figure](figsize=LEGEND_FIGURE_SIZE)
    for proposal_name in [PRIOR_PROPOSAL_NAME, OPTIMAL_PROPOSAL_NAME]
        for num_metainference in sort(unique(data[COL_NUM_METAINFERENCE]))
            if num_metainference == 1
                metainference_label = "$num_metainference meta-inference run (\$M_t = $num_metainference\$)"
            else
                metainference_label = "$num_metainference meta-inference runs (\$M_t = $num_metainference\$)"
            end
            label=latexstring("SMC, $(proposal_short_names[proposal_name]), $metainference_label")
            plt[:plot]([], [], linestyle=linestyles[num_metainference], color=colors[proposal_name], linewidth=2, label=label)
        end
    end
    plt[:axis]("off")
    plt[:legend](fontsize=14, ncol=1, frameon=false, borderpad=0, borderaxespad=0, loc="center")
    plt[:tight_layout](pad=0)
    plt[:savefig](fname)
end

# do plotting
data = readtable("$DATA_DIR/aide_estimates.csv")
plot_aide_estimate_exact_gold_standard(data, "$PLOT_DIR/divergences_exact_gold_standard.pdf")
plot_aide_estimate_approximate_gold_standard(data, "$PLOT_DIR/divergences_approximate_gold_standard.pdf")
plot_legend(data, "$PLOT_DIR/divergences_legend.pdf")
println("done!")
