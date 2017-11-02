using DataFrames
using Query
using PyPlot
using LaTeXStrings

include("schema.jl")
include("plotting.jl")

const LEGEND_FIGURE_SIZE = (3.2, 5)
const PLOT_FIGURE_SIZE = (3, 2.5)
const COLOR_LIKELIHOOD_WEIGHTING_ONE_PARTICLE = "white"
const COLOR_GOLD_STANDARD = "gold"
const ANNOTATION_FONTSIZE = 11
const TITLE_FONTSIZE=14

function annotate_num_clusters_plot(data::DataFrame)

    # draw filled circle for gold-standard
    gold_standard = @from i in data begin
            @where ((i.proposal_name == OPTIMAL_PROPOSAL_NAME) &&
                    (i.num_sweeps == 4) &&
                    (i.num_particles == 400))
            @select {i.num_particles, i.num_clusters_estimate}
            @collect DataFrame
        end
    add_filled_circle(gold_standard[COL_NUM_PARTICLES],
                      gold_standard[COL_NUM_CLUSTERS_ESTIMATE],
                      COLOR_GOLD_STANDARD)
    
    # draw horizontal dashed line for gold standard 
    xlim = get_xlim(data)
    gold_standard_value = gold_standard[COL_NUM_CLUSTERS_ESTIMATE]
    plt[:plot](xlim, [gold_standard_value, gold_standard_value], "k--")
end


function plot_num_clusters(data::DataFrame, fname::String, to_plot::Vector{Tuple{String,Int}})
    plt[:figure](figsize=PLOT_FIGURE_SIZE)
    for (proposal_name, num_sweeps) in to_plot
        result = @from i in data begin
            @where ((i.proposal_name == proposal_name) &&
                    (i.num_sweeps == num_sweeps))
            @select {i.num_particles, i.num_clusters_estimate, i.num_clusters_stderr}
            @collect DataFrame
        end
        plt[:plot](result[COL_NUM_PARTICLES], result[COL_NUM_CLUSTERS_ESTIMATE])
        lower = result[COL_NUM_CLUSTERS_ESTIMATE] - result[COL_NUM_CLUSTERS_STDERR] 
        upper = result[COL_NUM_CLUSTERS_ESTIMATE] + result[COL_NUM_CLUSTERS_STDERR]
        plt[:fill_between](result[COL_NUM_PARTICLES], lower, upper, alpha=0.3)
    end

    annotate_num_clusters_plot(data)

    ax = plt[:gca]()
    ax[:set_xscale]("log")
    ax[:set_xlim](get_xlim(data))
    plt[:ylabel]("Average number\nof clusters")
    plt[:xlabel]("Number of particles")
    plt[:title]("Heuristic diagnostic", fontsize=TITLE_FONTSIZE)
    plt[:tight_layout](pad=0)
    plt[:savefig](fname)
end

function annotate_aide_plot(data::DataFrame)

    # draw filled circle for gold-standard
    gold_standard = @from i in data begin
            @where ((i.proposal_name == OPTIMAL_PROPOSAL_NAME) &&
                    (i.num_sweeps == 4) &&
                    (i.num_particles == 400))
            @select {i.num_particles, i.aide_estimate}
            @collect DataFrame
        end
    add_filled_circle(gold_standard[COL_NUM_PARTICLES],
                      gold_standard[COL_AIDE_ESTIMATE],
                      COLOR_GOLD_STANDARD)

    # draw horizontal dashed line for gold standard 
    xlim = get_xlim(data)
    gold_standard_value = gold_standard[COL_AIDE_ESTIMATE]
    plt[:plot](xlim, [gold_standard_value, gold_standard_value], "k--")
end

function plot_aide(data::DataFrame, fname::String, to_plot::Vector{Tuple{String,Int}})
    plt[:figure](figsize=PLOT_FIGURE_SIZE)
    for (proposal_name, num_sweeps) in to_plot
        result = @from i in data begin
            @where ((i.proposal_name == proposal_name) &&
                    (i.num_sweeps == num_sweeps))
            @select {i.num_particles, i.aide_estimate, i.aide_stderr}
            @collect DataFrame
        end
        plt[:plot](result[COL_NUM_PARTICLES], result[COL_AIDE_ESTIMATE])
        lower = result[COL_AIDE_ESTIMATE] - result[COL_AIDE_STDERR] 
        upper = result[COL_AIDE_ESTIMATE] + result[COL_AIDE_STDERR]
        plt[:fill_between](result[COL_NUM_PARTICLES], lower, upper, alpha=0.3)
    end

    annotate_aide_plot(data)

    ax = plt[:gca]()
    ax[:set_xscale]("log")
    ax[:set_yscale]("log")
    ax[:set_xlim](get_xlim(data))
    plt[:ylabel]("nats")
    plt[:xlabel]("Number of particles")
    plt[:title]("AIDE estimates", fontsize=TITLE_FONTSIZE)
    plt[:tight_layout](pad=0)
    plt[:savefig](fname)
end

function plot_legend(data::DataFrame, fname::String, to_plot::Vector{Tuple{String,Int}})
    plt[:figure](figsize=LEGEND_FIGURE_SIZE)
    for (proposal, num_sweeps) in to_plot
        plt[:plot]([], [], label="SMC, $proposal proposal\n$num_sweeps rejuvenation sweep(s)")
    end
    plt[:scatter]([], [], color="gold", s=70, edgecolor="black", label="Gold-standard")
    plt[:legend](fontsize=13, ncol=1, frameon=false, borderpad=0, borderaxespad=0, loc="lower left", labelspacing=2)
    plt[:axis]("off")
    plt[:tight_layout](pad=0)
    plt[:savefig](fname)
end

function plot_histogram()
    data_full = readcsv("$data_dir/galaxy_velocities.txt")[:]
    plt[:figure](figsize=(5, 2))
    plt[:hist](data_full, bins=50)
    plt[:xlabel]("Recessional velocity (redshift, km/s)")
    plt[:gca]()[:set_xlim](0, 50000)
    plt[:tight_layout](pad=0)
    plt[:savefig]("$plot_dir/galaxies_histogram.pdf")
end


data_dir = "data"
data = readtable("$data_dir/galaxy_sampler_data.csv")

plot_dir = "plots"
if !Base.Filesystem.isdir(plot_dir)
    Base.Filesystem.mkdir(plot_dir)
end

# which proposals, and which number of sweeps to plot, as a function of number
# of particles
to_plot =Tuple{String,Int}[]
for proposal in (PRIOR_PROPOSAL_NAME, OPTIMAL_PROPOSAL_NAME)
    for num_sweeps in [0, 1, 4]
        push!(to_plot, (proposal, num_sweeps))
    end
end
plot_aide(data, "$plot_dir/galaxies_aide.pdf", to_plot)
plot_num_clusters(data, "$plot_dir/galaxies_num_clusters.pdf", to_plot)
plot_legend(data, "$plot_dir/galaxies_legend.pdf", to_plot)
plot_histogram()
