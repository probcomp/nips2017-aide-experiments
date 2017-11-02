using DataFrames
using Query
using PyPlot
using LaTeXStrings

include("schema.jl")
include("plotting.jl")

const LEGEND_FIGURE_SIZE = (3.2, 3.5)
const PLOT_FIGURE_SIZE = (3, 2.5)
const COLOR_LIKELIHOOD_WEIGHTING_ONE_PARTICLE = "white"
const COLOR_GOLD_STANDARD = "gold"
const ANNOTATION_FONTSIZE = 11
const TITLE_FONTSIZE=16


function annotate_num_clusters_plot(data::DataFrame)

    # draw filled circle for likelihood weighting with one particle
    likelihood_weighting_one_particle = @from i in data begin
            @where ((i.proposal_name == PRIOR_PROPOSAL_NAME) &&
                    (i.num_sweeps == 0) &&
                    (i.num_particles == 1))
            @select {i.num_particles, i.num_clusters_estimate}
            @collect DataFrame
        end
    add_filled_circle(likelihood_weighting_one_particle[COL_NUM_PARTICLES],
                      likelihood_weighting_one_particle[COL_NUM_CLUSTERS_ESTIMATE],
                      COLOR_LIKELIHOOD_WEIGHTING_ONE_PARTICLE)

    # draw filled circle for gold-standard
    gold_standard = @from i in data begin
            @where ((i.proposal_name == OPTIMAL_PROPOSAL_NAME) &&
                    (i.num_sweeps == 4) &&
                    (i.num_particles == 100))
            @select {i.num_particles, i.num_clusters_estimate}
            @collect DataFrame
        end
    add_filled_circle(gold_standard[COL_NUM_PARTICLES],
                      gold_standard[COL_NUM_CLUSTERS_ESTIMATE],
                      COLOR_GOLD_STANDARD)
    
    # draw arrow and text box for likelihood weighting with one particle
    head = (0.07, 0.69)
    base = (0.35, 0.87)
    draw_arrow(base[1], base[2], head[1] - base[1], head[2] - base[2])
    text(0.35, 0.9, "Appears accurate", ANNOTATION_FONTSIZE)

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

    # draw filled circle for likelihood weighting with one particle
    likelihood_weighting_one_particle = @from i in data begin
            @where ((i.proposal_name == PRIOR_PROPOSAL_NAME) &&
                    (i.num_sweeps == 0) &&
                    (i.num_particles == 1))
            @select {i.num_particles, i.aide_estimate}
            @collect DataFrame
        end
    add_filled_circle(likelihood_weighting_one_particle[COL_NUM_PARTICLES],
                      likelihood_weighting_one_particle[COL_AIDE_ESTIMATE],
                      COLOR_LIKELIHOOD_WEIGHTING_ONE_PARTICLE)
    
    # draw arrow and text box for likelihood weighting with one particle
    head = (0.07, 0.93)
    base = (0.23, 0.9)
    draw_arrow(base[1], base[2], head[1] - base[1], head[2] - base[2])
    text(0.25, 0.7, "Likelihood weighting\nwith 1 particle appears\nleast accurate", ANNOTATION_FONTSIZE)

    # draw horizontal dashed line for zero KL divergence
    xlim = get_xlim(data)
    plt[:plot](xlim, [0., 0.], "k--")
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
    ax[:set_xlim](get_xlim(data))
    plt[:ylabel]("nats")
    plt[:xlabel]("Number of particles")
    plt[:title]("AIDE estimates", fontsize=TITLE_FONTSIZE)
    plt[:tight_layout](pad=0)
    plt[:savefig](fname)
end

function plot_legend(data::DataFrame, fname::String)
    plt[:figure](figsize=LEGEND_FIGURE_SIZE)
    plt[:plot]([], [], label="SMC, prior proposal\n0 rejuvenation sweeps")
    plt[:plot]([], [], label="SMC, optimal proposal\n0 rejuvenation sweeps")
    plt[:plot]([], [], label="SMC, optimal proposal\n4 rejuvenation sweeps")
    plt[:scatter]([], [], color="gold", s=70, edgecolor="black", label="Gold-standard")
    plt[:scatter]([], [], color="white", s=70, edgecolor="black", label="Likelihood-weighting\n(1 particle)")
    plt[:legend](fontsize=13, ncol=1, frameon=false, borderpad=0, borderaxespad=0, loc="lower left", labelspacing=2)
    plt[:axis]("off")
    plt[:tight_layout](pad=0)
    plt[:savefig](fname)
end

data_dir = "data"

plot_dir = "plots"
if !Base.Filesystem.isdir(plot_dir)
    Base.Filesystem.mkdir(plot_dir)
end

data = readtable("$data_dir/sampler_data.csv")

# which proposals, and which number of sweeps to plot, as a function of number
# of particles
to_plot = [
    (PRIOR_PROPOSAL_NAME, 0),
    (OPTIMAL_PROPOSAL_NAME, 0),
    (OPTIMAL_PROPOSAL_NAME, 4)]
plot_aide(data, "$plot_dir/aide.pdf", to_plot)
plot_num_clusters(data, "$plot_dir/num_clusters.pdf", to_plot)
plot_legend(data, "$plot_dir/legend.pdf")

