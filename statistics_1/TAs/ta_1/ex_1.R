# Exercise 1

Claims <- c(rep(0, 21), rep(1,13), rep(2,5), rep(3,4), rep(4,2), rep(5,3), rep(6,2))

# (c) 
t <- table(Claims)
summary(Claims)

# mean > median => rx-skewed distribution
barplot(Claims)
boxplot(Claims)

# (d)
variance = var(Claims)
stdev = variance ** (0.5)
interval.95 = c(mean(Claims) - 2 * stdev, mean(Claims ) + 2 * stdev)
interval.95
percent.within.2stdev = 1 - length(Claims[Claims >= 5]) / length(Claims)

# => no, only 90% of obs lie within 2 standard deviations
stdev
variance
