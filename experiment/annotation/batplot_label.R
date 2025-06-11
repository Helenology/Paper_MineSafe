data = read.csv("/Users/helenology/Documents/GitHub/Paper_MineSafe/MineSafe-2024/experiment/annotation/label-0515.csv")
a = data$Label
ratio = table(a) / length(a)
print(ratio)
bp <- barplot(table(a) / length(a),
        ylim=c(0, 0.8),
        names.arg=c("Vacant", "Safe", "Unsafe"))
text(bp, ratio + 0.04, labels = c("64.3%", "20.2%", "15.5%"))
