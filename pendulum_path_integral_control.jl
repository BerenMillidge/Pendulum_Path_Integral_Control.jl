using LinearAlgebra
using PyCall
using Flux
using Statistics, StatsBase

GLU = pyimport("OpenGL.GLU") # needed for rendering some OpenAIGym environemnts
np = pyimport("numpy")
gym = pyimport("gym")

const timesteps = 20 #T
const nsamples = 1000 #K
const action_low = -2.0f0
const action_high = 2.0f0
const noise_mu = 0f0
const noise_sigma = 10f0
const lambda = 1f0

U = rand(Float32, timesteps) .* (2 * action_high) .+ action_low
env = gym.make("Pendulum-v0")
env.reset()
##

cost_total = zeros(Float32, nsamples)
function compute_total_cost(cost_total, k)
    env.env.state = x_init
    for t in 1:timesteps
        peturbed_action_t = U[t] + noise[k,t]
        _, r, _, _ = env.step([peturbed_action_t])
        cost_total[k] -= r
    end
    return cost_total
end
ensure_non_zero(cost, beta, factor) = exp.(-factor .* (cost .- beta))

function path_integral_MPC(U,env, x_init, noise)
    cost_total = zeros(Float32, nsamples)
    for k in 1:nsamples
        env.env.state = x_init
        for t in 1:timesteps
            peturbed_action_t = U[t] + noise[k,t]
            _, r, _, _ = env.step([peturbed_action_t])
            cost_total[k] -= r
        end
    end
    beta = minimum(cost_total,dims=1)
    cost_total_non_zero = ensure_non_zero(cost_total, beta, 1/lambda)
    eta = sum(cost_total_non_zero)
    omega = cost_total_non_zero ./ eta
    U= U .+  [sum(omega .* noise[:,t]) for t in 1:timesteps]
    return U
end

function run_planner(env, num_iterations)
    s = env.reset()
    x_init = env.env.state
    U = rand(Float32, timesteps) .* (2 * action_high) .+ action_low
    noise_gaussian = noise_mu .+ (randn(Float32,nsamples, timesteps) .* noise_sigma)
    noise_filled = ones(Float32, nsamples, timesteps) .* 0.9f0
    noise = noise_gaussian
    for iter in 1:num_iterations
        U = path_integral_MPC(U, env, x_init, noise)
        env.env.state = x_init # reset
        s,r,done,info = env.step([U[1]])
        println("action taken: $(U[1]), cost received: $(-r)")
        env.render()
        U = circshift(U, -1)
        U[end] = 1
        println("mean: $(mean(U))")
        x_init = env.env.state
    end
end

run_planner(env, 1000)
