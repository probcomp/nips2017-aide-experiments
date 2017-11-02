type NIGN
    N::Int
    sum_x::Float64
    sum_x_sq::Float64
    function NIGN()
        new(0, 0.0, 0.0)
    end
end

function incorporate!(comp::NIGN, x::Float64)
    comp.N += 1
    comp.sum_x += x
    comp.sum_x_sq += x*x
end

function unincorporate!(comp::NIGN, x::Float64)
    @assert comp.N > 0
    comp.N -= 1
    comp.sum_x -= x
    comp.sum_x_sq -= x*x
end

immutable NIGNParamsPrior
    m_prior::Normal
    r_prior::Gamma
    nu_prior::Gamma
    s_prior::Gamma
end

type NIGNParams
	m::Float64
	r::Float64
    nu::Float64
    s::Float64
    function NIGNParams(m::Float64, r::Float64, nu::Float64, s::Float64)
        new(m, r, nu, s)
    end
    function NIGNParams(prior::NIGNParamsPrior)
        m = rand(prior.m_prior)
        r = rand(prior.r_prior)
        nu = rand(prior.nu_prior)
        s = rand(prior.s_prior)
        new(m, r, nu, s)
    end
end

function log_probability_density(prior::NIGNParamsPrior, params::NIGNParams)
    l = logpdf(prior.m_prior, params.m)
    l += logpdf(prior.r_prior, params.r)
    l += logpdf(prior.nu_prior, params.nu)
    l += logpdf(prior.s_prior, params.s)
    l
end

function log_z(r::Float64, s::Float64, nu::Float64)
    lz = ((nu + 1.) / 2.) * log(2)
    lz += .5 * log(pi)
    lz -= .5 * log(r)
    lz -= (nu/2.) * log(s)
	lz += lgamma(nu/2.0)
    lz
end

function posterior_params(comp::NIGN, params::NIGNParams)
	rn = params.r + Float64(comp.N)
	nun = params.nu + Float64(comp.N)
	mn = (params.r*params.m + comp.sum_x)/rn
	sn = params.s + comp.sum_x_sq + params.r*params.m*params.m - rn*mn*mn
	if sn == Float64(0)
		sn = params.s
	end
	(mn, rn, sn, nun)
end

function log_probability_density(comp::NIGN, params::NIGNParams)
    (mn, rn, sn, nun) = posterior_params(comp, params)
    Z0 = log_z(params.r, params.s, params.nu)
    ZN = log_z(rn, sn, nun)
	-(comp.N/2.) * log(2 * pi) + ZN - Z0
end
