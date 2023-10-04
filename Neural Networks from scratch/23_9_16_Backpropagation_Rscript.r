#Project
library(ggplot2)
setwd("/Users/adamkurth/Documents/vscode/Python/Neural Networks from scratch/")
data = read.csv('output.csv')

print(data)
colnames(data) = NULL
colnames(data) = c("b1_mean", "b2_mean", "W1_mean", "W2_mean", "Accuracy", "Iteration")
accuracy = data$Accuracy
iteration = data$Iteration
mean.b1 = data$b1_mean
mean.b2 = data$b2_mean
mean.w1 = data$W1_mean
mean.w2 = data$W2_mean

plot(mean.w1~iteration, col = 'blue')
plot(mean.b1~iteration, col = 'red')


# GRAPH 3D GRADIENT DESCENT

log.accuracy = log(accuracy)
plot(iteration, log(accuracy))
log.lm = lm(log.accuracy ~ iteration)
abline(log.lm)

# here we see the accuracy increase for every iteration
ggplot(data, aes(x = iteration, y = accuracy)) +
  geom_point() +
  geom_line() +
  labs(title="Exponential Growth of Backpropagation Network",
   x = "Log(Accuracy)", y = "Iteration") +
  theme_minimal()


# can see the weights increasing at a significant increas, and not the bias term
ggplot(data, aes(x = iteration)) +
  geom_line(aes(y = mean.w1), color = "blue", size = 1) +
  geom_line(aes(y = mean.b1), color = "red", size = 1) +
  
  # Set plot labels and title
  labs(
    x = "Iteration",
    y = "Mean Value",
    title = "Mean Values Over Iteration"
  ) +
  
  # Adjust the appearance (optional)
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "top"
  )
