Errors = c(rep(0,102), rep(1,138), rep(2,140), rep(3,79), rep(4,33), rep(5,8))
abs.freq = table(Errors)
rel.freq = prop.table(abs.freq)
cum.sum = cumsum(rel.freq)

t = rbind(abs.freq, rel.freq,cum.sum)
t

summary(Errors) # seems left skewed NO - no clear symmetry emerges
boxplot(Errors)
barplot(Errors, col='red')
