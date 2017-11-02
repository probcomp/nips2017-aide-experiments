using Base.Test

include("dpmm.jl")
include("smc_samplers.jl")

@testset "CRP tests" begin

    @testset "assignment book-keeping" begin
        crp = CRPState()

        incorporate!(crp, 1, next_new_cluster(crp))
        c1 = assignment(crp, 1)
        @test c1 == 1

        incorporate!(crp, 2, next_new_cluster(crp))
        c2 = assignment(crp, 2)
        @test c2 == 2

        incorporate!(crp, 3, c2)
        @test counts(crp, c1) == 1
        @test counts(crp, c2) == 2
        @test next_new_cluster(crp) == c2 + 1
        @test crp.next_cluster == c2 + 2

        unincorporate!(crp, 1)
        @test !has_cluster(crp, c1)
        @test counts(crp, c2) == 2
        @test next_new_cluster(crp) == c1

        unincorporate!(crp, 2)
        @test counts(crp, c2) == 1

        unincorporate!(crp, 3)
        @test !has_cluster(crp, c2)
        @test next_new_cluster(crp) == c2

        # this is never decremented
        @test crp.next_cluster == c2 + 2
    end

    @testset "log_probability" begin

        alpha = 0.25
        crp = CRPState()
        @test log_probability(crp, alpha) == 0.0

        # [1]
        incorporate!(crp, 1, next_new_cluster(crp))
        @test log_probability(crp, alpha) == 0.0
        
        # join the same table as 1: [1,2]
        incorporate!(crp, 2, assignment(crp, 1))
        actual = log_probability(crp, alpha)
        expected = log(1. / (1. + alpha))
        @test isapprox(actual, expected)

        # join a new table: [1] [2]
        unincorporate!(crp, 2)
        incorporate!(crp, 2, next_new_cluster(crp))
        actual = log_probability(crp, alpha)
        expected = log(alpha / (1. + alpha))
        @test isapprox(actual, expected)

        # join same table as 1: [1, 3] [2]
        incorporate!(crp, 3, assignment(crp, 1))
        actual = log_probability(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)))
        @test isapprox(actual, expected)

        # join same table as 1 [1, 3, 4] [2]
        incorporate!(crp, 4, assignment(crp, 1))
        actual = log_probability(crp, alpha)
        expected = log((alpha / (1.+alpha)) * 
                       (1./(2.+alpha)) *
                       (2./(3.+alpha)))
        @test isapprox(actual, expected)
    end

end

@testset "NIGN" begin

    # arbitrary values
    m = 1.23
    r = 2.41
    nu = 5.44
    s = 4.2
    alpha = nu/2.
    beta = s/2.
    params = NIGNParams(m, r, nu, s)

    nign = NIGN()
    actual = log_probability_density(nign, params)
    @test abs(actual) < 1e-15

    # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    x1 = -4.1
    x2 = 2.22
    incorporate!(nign, x1)
    incorporate!(nign, x2)
    (mu_actual, r_actual, s_actual, nu_actual) = posterior_params(nign, params)
    # m, r, s, nu
    avg = mean([x1, x2])
    sum_sq = sum(([x1, x2] - avg) .* ([x1,x2] - avg))
    n = 2
    mu_expect = (r * m + n * avg) / (r + n)
    r_expect = r + n
    alpha_expect = alpha + n/2.
    beta_expect = beta + 0.5 * sum_sq  + r*n*((avg-m)*(avg-m))/(2*(r+n))
    nu_expect = alpha_expect*2.
    s_expect = beta_expect*2.
    @test isapprox(mu_actual, mu_expect)
    @test isapprox(r_actual, r_expect)
    @test isapprox(s_actual, s_expect)
    @test isapprox(nu_actual, nu_expect)

    # should go back to empty again
    unincorporate!(nign, x1)
    unincorporate!(nign, x2)
    actual = log_probability_density(nign, params)
    @test abs(actual) < 1e-15

end

@testset "DPMM SMC" begin

    @testset "extreme easy two-cluster dataset" begin
        srand(1)

        # generate dataset
        num_data = 50
        data = Array{Float64,1}(num_data)
        cluster_1_members = []
        cluster_2_members = []
        for i=1:num_data
            if rand() < 0.5
                mu = -1.0
                push!(cluster_1_members, i)
            else
                mu = 1.0
                push!(cluster_2_members, i)
            end
            std = 0.01
            data[i] = rand(Normal(mu, std))
        end

        # set custom values of hypers to match extreme dataset
        m_prior = Normal(0, 1) # mean cluster center 
        r_prior = Gamma(1, 1) # r * data-precision is the precision of cluster centers
        nu_prior = Gamma(1, 1)# expected data-precision of cluster is nu/s
        s_prior = Gamma(1, 1)
        nign_params_prior = NIGNParamsPrior(m_prior, r_prior, nu_prior, s_prior)
        alpha_prior = Gamma(1, 1)
        hypers = Hyperparameters(nign_params_prior, alpha_prior)

        gibbs_sweeps_per_datum = 1
        num_particles = 1
        scheme = make_optimal_proposal_smc_scheme(data, hypers, 
                               gibbs_sweeps_per_datum, num_particles)
        state, lml_estimate = smc(scheme)

        # check that there are two clusters, and that clustering
        # is correct
        @test length(state.components) == 2
        cluster_1 = assignment(state.crp, cluster_1_members[1])
        cluster_2 = assignment(state.crp, cluster_2_members[2])
        @test cluster_1 != cluster_2
        for i in cluster_1_members[2:end]
            @test assignment(state.crp, i) == cluster_1
        end
        for i in cluster_2_members[2:end]
            @test assignment(state.crp, i) == cluster_2
        end
        
    end

    @testset "prior proposal weight" begin
        alpha = 0.01
        m = 0.9
        r = 1.1
        nu = 1.2
        s = 1.3
        nign_params = NIGNParams(m, r, nu, s)
        alpha_prior = Gamma(1, 1)
        nign_params_prior = NIGNParamsPrior(Normal(0, 1), Gamma(1, 1), Gamma(1, 1), Gamma(1, 1))
        hypers = Hyperparameters(nign_params_prior, alpha_prior)
	    state = DPMMState(alpha, CRPState(), OrderedDict{Int,NIGN}(), 
                      nign_params, hypers)

        srand(1)

        x1 = 4.2
        ll1 = add_datum_prior!(state, 1, x1)
        ll1_rev = remove_datum_prior!(state, 1, x1)
        @test isapprox(ll1, ll1_rev)

        x2 = -3.1
        ll1_new = add_datum_prior!(state, 1, x1)
        @test isapprox(ll1, ll1_new)
        ll2 = add_datum_prior!(state, 2, x2)
        # NOTE: we seeded so this should remain true, if it doesn't change the
        # seed or make alpha even smaller
        @test assignment(state.crp, 1) == assignment(state.crp, 2)
        component = NIGN()
        incorporate!(component, x1)
        incorporate!(component, x2)
        total_logp = log_probability_density(component, nign_params)
        @test isapprox(total_logp, ll1_new + ll2)

        ll1_rev = remove_datum_prior!(state, 1, x1)
        ll2_rev = remove_datum_prior!(state, 2, x2)
        @test isapprox(total_logp, ll1_rev + ll2_rev)
    end

    @testset "optimal proposal weight" begin
        alpha = 2.45
        m = 0.9
        r = 1.1
        nu = 1.2
        s = 1.3
        nign_params = NIGNParams(m, r, nu, s)
        alpha_prior = Gamma(1, 1)
        nign_params_prior = NIGNParamsPrior(Normal(0, 1), Gamma(1, 1), Gamma(1, 1), Gamma(1, 1))
        hypers = Hyperparameters(nign_params_prior, alpha_prior)
	    state = DPMMState(alpha, CRPState(), OrderedDict{Int,NIGN}(), 
                      nign_params, hypers) 

        srand(1)

        x1 = 4.2
        ll1 = add_datum_conditional!(state, 1, x1)
        ll1_rev = remove_datum_conditional!(state, 1, x1)
        @test isapprox(ll1, ll1_rev)
        component = NIGN()
        incorporate!(component, x1)
        @test isapprox(ll1, log_probability_density(component, nign_params))

        ll1_new = add_datum_conditional!(state, 1, x1)
        x2 = 1.5
        @test isapprox(ll1, ll1_new)
        ll2 = add_datum_conditional!(state, 2, x2)
        ll2_rev = remove_datum_conditional!(state, 2, x2)
        @test isapprox(ll2, ll2_rev)

        # explicitly sum over the two possible assignments
        # p(x_2 = alone | x_1) p(y_2 | x_1, x_2 = alone, y_1)
        component_alone = NIGN()
        incorporate!(component_alone, x2)
        ll2_alone = log_probability_density(component_alone, nign_params)
        ll2_alone += log(alpha / (1. + alpha))

        # p(x_2 = together | x_1) p(y_2 | x_1, x_2 = together, y_1)
        component_together = NIGN() 
        incorporate!(component_together, x1)
        component_old_logpdf = log_probability_density(component_together, nign_params)
        incorporate!(component_together, x2)
        ll2_together = log_probability_density(component_together, nign_params) - component_old_logpdf
        ll2_together += log(1. / (1. + alpha))

        # p(y_2 | x_1, y_1) = sum of two terms above
        ll2_expected = logsumexp([ll2_alone, ll2_together])
        @test isapprox(ll2, ll2_expected)
    end

end
