abs.freq.region = table(DW$region)
pie(abs.freq.region)

abs.freq.education = table(DW$education)
abs.freq.education
barplot(abs.freq.education, col='green', cex.names=0.5)

wage.summ <- summary(DW$wage)
ed.summ <- summary(DW$education)

vardata <- DW$wage
wage.iqr <- wage.summ['3rd Qu.'] - wage.summ['1st Qu.']
wage.quants <- quantile(vardata, c(0.2,0.95))
wage.var <- var(vardata)
wage.stdev <- wage.var ** (0.5)

vardata <- DW$education
ed.iqr <- ed.summ['3rd Qu.'] - ed.summ['1st Qu.']
ed.quants <- quantile(vardata, c(0.2,0.95))
ed.var <- var(vardata)
ed.stdev <- ed.var ** (0.5)
correlation <- cor(DW$wage, DW$education)
correlation

# (e)
poorest.10 <- quantile(DW$wage, 0.1)
poorest.10 # salary threshold is 182.1
n_workers <- length(DW$wage[DW$wage <= poorest.10])
n_workers # 2817 workers

# (f)
boxplot(DW$wage) # right skewed

# (g)
hist(DW$wage, breaks = 20)
logwage = log10(DW$wage)
hist(logwage, breaks=20)

# (j)
tapply(DW$wage, DW$smsa, mean)
boxplot(logwage~DW$smsa)
