# we saw that we set xt+1= xt + alpha * pt

# pt is just - gradient
# how can we optimize alpha?
# we can use a line search procedure

# we look at a slice of the function taht contains the gradient vector
# we now have to find the min of a 2d function (eg use newton)
# now we get to the min point of the 3d function along that line
# we repeat the procedure and now the next step will be orthogonal

# we can express
# xt+1 = xt - (Bt)^-1 * gradient(f(xt))
# 1) Bt is identity matrix / alpha => we move along the gradient (gradient descent)
#   a) if B is not identity we will move by a rotation of the gradient
# 2) Bt = Hessian(f(xt)) => newton method
#   a) computing the inverse of hessian is O(n^3)
# 3) we now look at another way: quasi newton methods
#   a) usually hessian has rank n
#   b) we don't want to calculate all the vectors (O(n^2))
#   c) we want to find a rank 1 approximation of the hessian
#   d) we can do that by finding some conditions that Bt has to satisfy
#       in order to be similar to the hessian
#   e) we use the taylor theorem: f(x + p) = f(xt) + gradient(f(xt)) * p + 1/2 p^T * Hessian(f(xt)) * p + O(|p|^3)
#   f) we want to find a matrix Bt that satisfies the following conditions: f(x + p) = f(xt) + gradient(f(xt)) * p + 1/2 p^T * Bt * p + O(|p|^2)
#   g) we set some conditions: (1) f(p=0) = f(xt), (2) gradf(p=0) = f(xt), (3) gradf(p = -step^(t-1) (last step we took)) = gradf(x - step^(t-1))
#       point (2): gradf(x+p) = (differentiating) gradf(x) + H(xt) * p. if p = 0 then we have pt (2)

#  4) momentum methods

# xt+1 = xt + pt
# we also write an equation for pt
# p_t = beta * p_t-1 - alpha * gradf(xt)
# 1) beta = 0 => grad descx
# 2) beta > 0 (he said beta > 1) => memory of previous velocities => gradient descent with momentum
# usual value if 0.9, used a lot in ml because you want to go down as fast as you can

# nesterov momentum
# xt+1 = xt + pt
# p_t = beta * p_t-1 - alpha * gradf(xt + beta * p_t-1)
# you eval the grad at a displaced point modified by the previous velocity
# this way you also keep momentum in the direction of the vector, not only the magnitude
# intuitively: we look at what would be the movement with no gradient (beta * pt-1) and eval the gradient there, then sum them

# how can we generalize this to multiple dimensions?