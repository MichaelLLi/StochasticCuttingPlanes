using Distributions, JuMP, Gurobi, DataFrames, Random, Statistics, LinearAlgebra

c = 4
k = 10
n = 1000000
q = 20
ntrials = 20
function Cutting_plane(W,q,x_cur;fast = false)
    n = size(W)[2]
    k = size(W)[1]
    if fast
      nnew = Int(round(sqrt(n) * 10))
    else
      nnew = n
    end
    samplen = sample(1:n, nnew, replace = false)
    Wsamp = W[:,samplen]
    objpart =  (Wsamp' * x_cur .- q)
    inds = findall(objpart .>= 0)
    obj =  c / nnew * sum(max.(objpart,0))
    if length(inds)>0
      ∇obj = c / nnew * sum(Wsamp[:,j] for j in inds)
    else
      ∇obj = zeros(k)
    end
    return obj, ∇obj
end

obj = zeros(ntrials, 3)
timing = zeros(ntrials, 3)
for trial = 1:ntrials
  r = rand(Uniform(10,20),k)
  mu = rand(Uniform(20,30),k)
  sigma = rand(Uniform(5,15),k)
  Σ = diagm(0 => sigma)
  W = rand(MvNormal(mu, Σ),n)

  # sskp = Model(Gurobi.Optimizer)
  # @variable(sskp, x[1:k],Bin)
  # @variable(sskp, z[1:n]>=0)
  # @constraint(sskp, [j=1:n], z[j]>= dot(W[:,j],x) - q)
  # @objective(sskp, Max, dot(r,x) - c / n * sum(z[j] for j=1:n))
  # time_result = @timed optimize!(sskp)
  # timing[trial,1] = time_result[2]
  # xopt1 = value.(x)
  # obj[trial,1] = dot(r,xopt1) - c / n * sum(max.(W' * xopt1 .- q, 0))

  global sskp2 = Model(Gurobi.Optimizer)
  @variable(sskp2, x[1:k],Bin)
  @variable(sskp2, t>=0)
  function Newcut(cb)
    x_cur = [callback_value(cb, x[i]) for i=1:k]
    obj, ∇obj = Cutting_plane(W,q,x_cur,fast=false)
    # add the cut: t >= obj + sum(∇s * (s - s_val))
    con = @build_constraint(t >= obj + sum(∇obj[j] * (x[j] - x_cur[j]) for j=1:k))
    MOI.submit(sskp2, MOI.LazyConstraint(cb), con)
  end
  MOI.set(sskp2, MOI.LazyConstraintCallback(), Newcut)
  @objective(sskp2, Max, dot(r,x) - t)
  time_result = @timed optimize!(sskp2)
  timing[trial,2] = time_result[2]
  xopt2 = value.(x)
  obj[trial,2] = dot(r,xopt2) - c / n * sum(max.(W' * xopt2 .- q, 0))


  global sskp3 = Model(Gurobi.Optimizer)
  @variable(sskp3, x[1:k],Bin)
  @variable(sskp3, t>=0)
  function Newcut2(cb)
    x_cur = [callback_value(cb, x[i]) for i=1:k]
    obj, ∇obj = Cutting_plane(W,q,x_cur,fast=true)
    # add the cut: t >= obj + sum(∇s * (s - s_val))
    con = @build_constraint(t >= obj + sum(∇obj[j] * (x[j] - x_cur[j]) for j=1:k))
    MOI.submit(sskp3, MOI.LazyConstraint(cb), con)
  end
  MOI.set(sskp3, MOI.LazyConstraintCallback(), Newcut2)
  @objective(sskp3, Max, dot(r,x) - t)
  time_result = @timed optimize!(sskp3)
  timing[trial,3] = time_result[2]
  xopt3 = value.(x)
  obj[trial,3] = dot(r,xopt3) - c / n * sum(max.(W' * xopt3 .- q, 0))
end
