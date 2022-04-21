using DataFrames, JuMP,Gurobi, StatsBase, LinearAlgebra, Distributions, CSV, LIBSVM

function SVMOpt(X,Y,C;ϵ=1e-4,fast=false)
n = size(X)[1]
p = size(X)[2]
m1 = Model(
    optimizer_with_attributes(
        Gurobi.Optimizer, "OutputFlag" => 0
    )
)
# Add variables
@variable(m1, w[1:p])
@variable(m1,t >= 0)
# Add constraints
@objective(m1, Min, 1/2 * dot(w, w) + C * t)
# solve initial
optimize!(m1)
wopt = value.(w)
topt = value(t)
nite = 0
offset, ∇obj = Cutting_plane(X,Y,wopt, fast = fast)
nite = nite + 1
while offset + dot(∇obj, wopt) > topt + ϵ
    @constraint(m1, t >= offset + dot(∇obj, w))
    optimize!(m1)
    wopt = value.(w)
    topt = value(t)
    offset, ∇obj = Cutting_plane(X,Y,wopt, fast = fast)
    nite = nite + 1
end
return wopt, topt, nite
end

function Cutting_plane(X,Y,w;fast=false)
    n = size(X)[1]
    if fast
        nnew = Int(round(sqrt(n))) * 10
    else
        nnew = n
    end
    samplen = sample(1:n, nnew, replace = false)
    c = (((X[samplen,:] .* Y[samplen]) * w) .< 1)
    offset =  mean(c)
    ∇obj = - mean(X[samplen,:] .* Y[samplen] .* c,dims=1)'
    return offset, ∇obj
end
df = CSV.File("data/covtype.data",header=false) |> DataFrame


n = 1000
samplen = sample(1:nrow(df), n, replace=false)
remaining = setdiff(1:nrow(df),samplen)
samplen2 = sample(remaining, n, replace=false)
train_X = Array(df[samplen,1:54])

train_Y = Array(df[samplen,55] .== 2) .* 2 .- 1
C = 1e6
t1 = time_ns()
wopt, topt, nite=SVMOpt(train_X,train_Y,C,fast=true)
Int(round(sqrt(n))) * 10 * nite
println((time_ns()-t1)/1e9)
println((time_ns()-t1)/1e9)

test_X = Array(df[samplen2,1:54])
test_Y = Array(df[samplen2,55] .== 2) .* 2 .- 1
pred_Y = sign.(test_X * wopt)
mean(pred_Y .== test_Y)

t1 = time_ns()
model = svmtrain(train_X', train_Y,cost = 100000.0, tolerance = 0.0001);
println((time_ns()-t1)/1e9)
println((time_ns()-t1)/1e9)
(pred_Y, decision_values) = svmpredict(model, test_X');
mean(pred_Y .== test_Y)
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
