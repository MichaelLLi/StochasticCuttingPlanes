using DataFrames, JuMP,Gurobi, StatsBase, LinearAlgebra, Distributions

function SparseRegOpt(X,Y,k,γ;fast=false)
n = size(X)[1]
p = size(X)[2]
m1 = Model(
    optimizer_with_attributes(
        Gurobi.Optimizer, "OutputFlag" => 1, "LazyConstraints" => 1, "Heuristics" => 0
    )
)
# Add variables
@variable(m1, z[1:p], Bin)
@variable(m1,t >= 0)
# Add constraints
@constraint(m1, sum(z[i] for i = 1:p) <= k)
@objective(m1, Min, t)
# z0=zeros(p)
# samplek = sample(1:p, k, replace = false)
# z0[samplek].=1
z0 = warmstart(X,Y,k,γ,fast = fast)
obj0, ∇obj0 = Cutting_plane(X,Y,z0,k,γ,fast = fast)
@constraint(m1, t >= obj0 + dot(∇obj0, z - z0))
# Outer approximation method for Convex Integer Optimization (CIO)
function Newcut(cb)
  z_cur = [callback_value(cb, z[i]) for i=1:p]
  obj, ∇obj = Cutting_plane(X,Y,z_cur,k,γ,fast = fast)
  # add the cut: t >= obj + sum(∇s * (s - s_val))
  con = @build_constraint(t >= obj + sum(∇obj[j] * (z[j] -z_cur[j]) for j=1:p))
  MOI.submit(m1, MOI.LazyConstraint(cb), con)
end
MOI.set(m1, MOI.LazyConstraintCallback(), Newcut)
println("Model Setup Complete")
# Solve the model and get the optimal solutions
optimize!(m1)
zopt = value.(z)
println("Model Solved")
Ypred = X[:,zopt.>0.5] * (inv(X[:,zopt.>0.5]' * X[:,zopt.>0.5]) * (X[:,zopt.>0.5]' * Y))
return Ypred, zopt
end

function Cutting_plane(X,Y,z0,k,γ;fast=false)
    n = size(X)[1]
    p = size(X)[2]
    # nnew = n
    if fast
        nnew = Int(round(Int(round(sqrt(n))) * 10))
    else
        nnew = n
    end
    samplen = sample(1:n, nnew, replace = false)
    Xsamp =  X[samplen, z0.>0.5]
    alpha = Y[samplen] - Xsamp * (inv(I / γ + Xsamp' * Xsamp) * (Xsamp' * Y[samplen]))
    obj = dot(alpha, Y[samplen]) / (2 * nnew)
    # appobj = sum((Y - X[:,z0.>0.5]* inv(X[:,z0.>0.5]'*X[:,z0.>0.5]) * X[:,z0.>0.5]'*Y).^2) / (2 * nnew)
    ∇obj = -γ * ((X[samplen, :]' * alpha) .^ 2) / (2 * nnew)
    # ∇obj = ∇obj .* (z0.>0.5) .* 2
    # println("The objective is $obj")
    # # println("The approxiamte objective is $appobj")
    # # println("The solution is $z0")
    # println("The derivative is $m∇obj")
    return obj, ∇obj
end
function warmstart(X,Y,k,γ;fast=false)
    n, p = size(X)

    # Saddle point algorithm
    iter = 0
    indices = []
    indices_new = 1:k
    if fast
        nnew = Int(round(Int(round(sqrt(n))) / 5))
    else
        nnew = n
    end
    output = zeros(p)
    while (iter < 10 || indices != indices_new && iter <= 100)
        samplen = sample(1:n, nnew, replace = false)
        iter += 1
        indices = indices_new

        # Maximize over α for a given s
        α_new = @views Y[samplen] .- X[samplen,indices] * (inv(I(k) ./ γ .+ X[samplen,indices]' * X[samplen,indices]) * (X[samplen,indices]' * Y[samplen]))
        output = (output .* (iter - 1) .+  abs.(X[samplen,:]' * α_new)) ./ iter
        # Maximize over s for a given α
        indices_new = sort(sortperm(output, rev=true)[1:k])
    end
    z_warm = zeros(p)
    z_warm[indices_new] .= 1
    return z_warm
end
n = 1000
p = 100
k = 5
γ = 100
X0 = rand(Normal(0,1),n,k)
Z = rand(Normal(0,1),n,p-k)
X = hcat(Z,X0)
β0 = rand(Normal(0,1),k)
Y = X0 * β0
n = size(X)[1]
p = size(X)[2]
Ypred, zopt=@time SparseRegOpt(X,Y,k,γ,fast=false)
median(abs.(Ypred .- Y)./abs.(Y))
avgacc = zeros(10)
for i = 1:10
    n = 1000
    p = 100
    k = 10
    γ = 0.1
    X0 = rand(Normal(0,1),n,k)
    Z = rand(Normal(0,1),n,p-k)
    X = hcat(Z,X0)
    β0 = rand(Normal(0,1),k)
    Y = X0 * β0 + rand(Normal(0,0.1),n)
    n = size(X)[1]
    p = size(X)[2]

    # t1 = time_ns()
    # Ypred, zopt=@time SparseRegOptDirect(X,Y,k,γ)

    Ypred, zopt=@time SparseRegOpt(X,Y,k,γ,fast=false)
    median(abs.(Ypred .- Y)./abs.(Y))

    zopt2 = zeros(p)
    zopt2[(p-k+1):p] .= 1
    avgacc[i] = mean((zopt2 .> 0.5).==(zopt .> 0.5))
end
# println((time_ns()-t1)/1e9)
median(abs.(Ypred .- Y)./abs.(Y))

# samplen = sample(1:n, n, replace = false)
# indices = sample(1:p, k, replace = false)
# @time Z = Y * 2
# @time  Z = @views Y[samplen] * 2
# @time Z = Y[samplen] * 2
# nnew = n
# samplen = sample(1:n, nnew, replace = false)
# z0=zeros(p)
# samplek = sample(1:p, k, replace = false)
# z0[samplek].=1
# Xsamp =  X[samplen, z0.>0.5]
# γ = 10
# obj0 = 1 / (2 * n) * Y' * inv(I + γ * sum(z0[i] * X[:,i]*X[:,i]' for i=1:p)) * Y
# z1 = copy(z0)
# z1[1] = z0[1] + 0.0000001
# obj1 = 1 / (2 * n) * Y' * inv(I + γ * sum(z1[i] * X[:,i]*X[:,i]' for i=1:p)) * Y
# alpha = Y - X[:,z0.>0.5] * (inv(I / γ + X[:,z0.>0.5]' * X[:,z0.>0.5]) * (X[:,z0.>0.5]' * Y))
# obj = dot(alpha, Y) / (2 * nnew)
# ∇obj = -γ * ((X' * alpha) .^ 2) / (2 * n)
# println((obj0 - obj))
# println((obj1 - obj0) / 0.0000001)
# println(∇obj[1])
#
# alpha = Y[samplen] - Xsamp * (inv(I / γ + Xsamp' * Xsamp) * (Xsamp' * Y[samplen]))
# obj = dot(alpha, Y[samplen]) / (2 * nnew)
# # appobj = sum((Y - X[:,z0.>0.5]* inv(X[:,z0.>0.5]'*X[:,z0.>0.5]) * X[:,z0.>0.5]'*Y).^2) / (2 * nnew)
# ∇obj = -γ * ((X[samplen, :]' * alpha) .^ 2) / (2 * nnew)
#
