rm(list=ls())
setwd("/Users/adamkurth/Documents/vscode/Python/linear_regression_live-master/")
data = read.csv('output_grad.csv')
weight = data[,1]
bias = data[,2]
error = data[,3]
