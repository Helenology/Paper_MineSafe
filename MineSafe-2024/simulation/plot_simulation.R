library(plyr)
library(ggplot2)
library(reshape2)

test_size = 100


# data = read.csv('/Users/helenology/Desktop/光华/ 论文/2-MineSafe/Simulation/0619-GPU1-experiment_stats_N=1000.csv')
data1 = read.csv("/Users/helenology/Documents/GitHub/Paper_MineSafe/MineSafe-2024/simulation/simulation(N=100).csv")
data2 = read.csv("/Users/helenology/Documents/GitHub/Paper_MineSafe/MineSafe-2024/simulation/simulation(N=500).csv")
data3 = read.csv("/Users/helenology/Documents/GitHub/Paper_MineSafe/MineSafe-2024/simulation/simulation(N=1000).csv")
data = rbind(data1, data2, data3)
data = data[, -c(1, 3)]


##################### MSE Comparison ##################### 
MSE_data = data[, c("N", "CD_mse", "DS_mse", "GPA_CD_mse", "GPA_DS_mse")]
MSE_data$N = as.factor(MSE_data$N)
MSE_data = melt(MSE_data, id=c("N"), 
                variable.name = "Estimator", 
                value.name = "MSE")
MSE_data$Estimator = factor(MSE_data$Estimator, 
                            levels = c('CD_mse', 'DS_mse', 'GPA_CD_mse', 'GPA_DS_mse'),
                            labels = c("CD", "DS", "GPA-CD", "GPA-DS"))
str(MSE_data)
# pmse = ggplot(dat1, aes(x = N, y = log(MSE), fill = Type)) +
#   stat_boxplot(geom = "errorbar",
#                width=0.3,
#                position = position_dodge(0.9)) +
#   geom_boxplot(position = position_dodge(0.9)) +
#   theme_test(base_size = 15) +
#   guides(fill=guide_legend(title=NULL)); pmse
#   

par(mfrow = c(1, 3))
plot_box = function(thres, times){
  boxplot(log(MSE)~Estimator, MSE_data[MSE_data$N == thres, ], 
          xlab="", cex.axis = times,
          cex.lab = times,
          ylim = c(-5.5, -1.5),
          yaxt ="n") # 统一y轴坐标
  axis(side = 2, # 操作y轴
       at = seq(-5.5, -1.5, 0.5),
       labels = seq(-5.5, -1.5, 0.5)
       )

  title(paste0('N=', thres), 
        line=0.7,   # 标题的位置
        font.main=1 # 标题取消加粗
        )
}
plot_box(100, 1.1)
plot_box(500, 1.1)
plot_box(1000, 1.1)

# boxplot(log(MSE)~Type, dat1[dat1$N == 500, ], xlab="")
# boxplot(log(MSE)~Type, dat1[dat1$N == 1000, ], xlab="")



# library(gridExtra)
# grid.arrange(p1,p2,nrow=1)
# 
# ggsave('/Users/helenology/Desktop/log_MSE_boxplot.png',
#        pmse,
#        width=18,
#        height=12,
#        dpi=800,
#        units="cm")
  
