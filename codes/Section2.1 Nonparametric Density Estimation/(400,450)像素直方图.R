library(ggplot2)
setwd("/Users/helenology/Documents/GitHub/Paper_MineSafe/codes/Section2.1\ Nonparametric\ Density\ Estimation/")
getwd()

dat1 = read.csv('./pixel_TS400450.csv',
                stringsAsFactors = F,
                header = F)
dat2 = read.csv('./pixel_TS445.csv',
                stringsAsFactors = F,
                header = F)
dat = rbind(dat1, dat2)

m = mean(dat$V1)
s = sd(dat$V1)
# c = pnorm((1-m)/s) - pnorm((0-m)/s)
# y = y / c
# dat0 = data.frame(x, y, label='label')

dat1 = dat[abs(dat$V1 - m) / s <= 2, ]
m = mean(dat1$V1)
s = sd(dat1$V1)
x = seq(0, 1, 0.001)
y = exp(-(x - m)**2 / (2 * s**2)) / (sqrt(2 * pi) * s) 
dat0 = data.frame(x, y)


hist(dat1$V1, breaks = 80, xlab = "", main = "")
par(new = TRUE)
plot(dat0$x, dat0$y, type ="l",
     xaxt = "n", yaxt = "n",
     ylab = "", xlab = "")
axis(side = 4)

h <- hist(dat1$V1, breaks = 80, plot=FALSE)
h$counts=h$counts/sum(h$counts)
plot(h, xlab = "", ylab="", main = "", col = 'lightgray')
par(new = TRUE)
plot(dat0$x, dat0$y, type ="l",
     xaxt = "n", yaxt = "n",
     ylab = "", xlab = "")
axis(side = 4)

# phist = ggplot(dat0, aes(x = x, y = y/8, color = label)) +
#   scale_color_manual(values=c('black'), labels = c('Fitted Truncated Normal Density Function')) +
#   scale_size_manual(values=c(150))+
#   geom_histogram(aes(x = V1, y = ..count../length(dat[, 1])), dat, fill = 'steelblue', color = 'black', binwidth = 0.05) + 
#   geom_line(size=0.8) + # linetype="dashed", 
#   theme_classic(base_size = 30) + 
#   theme(legend.position="top", legend.title=element_blank()) + 
#   scale_y_continuous(breaks = seq(0, 1, 0.2), sec.axis = sec_axis(~.*8,
#                                                                   name = '',
#                                                                   breaks = seq(0,5,1))) + 
#   labs(x = '', y = 'Frequency'); phist
# 
phist = ggplot() +
  # scale_color_manual(values=c('black'), labels = c('Fitted Truncated Normal Density Function')) +
  # scale_size_manual(values=c(150))+
  geom_histogram(aes(x = V1, y = ..count../length(dat[, 1])), 
                 dat1, 
                 fill = 'white',
                 color = 'black', 
                 bins=80) +
  theme_classic(); phist
  # geom_line(size=0.8) + # linetype="dashed",
  # theme_classic(base_size = 30) +
  # theme(legend.position="top", legend.title=element_blank()) +
  # scale_y_continuous(breaks = seq(0, 1, 0.2), sec.axis = sec_axis(~.*8,
  #                                                                 name = '',
  #                                                                 breaks = seq(0,5,1))) +
  labs(x = '', y = 'Frequency'); phist


ggsave('/Users/helenology/Desktop/400_hist.png',
       phist,
       width=30,
       height=20,
       dpi=800,
       units="cm")
