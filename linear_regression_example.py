###-------------------------------------------------------------------------###
###-------------- Example Python Code for Linear Regression ----------------###
###-------------------------------------------------------------------------###
##                                                                            #
#   This code is an example of how a line can be fit to a discrete [x,y]      #
#   dataset. This general method can be applied to any dataset in an          #
#   (x,y)-plane for which a line fit is appropriate.                          #
#                                                                             #
#   To run this code:                                                         #
#                                                                             #
#      1) Download and Install Anaconda (www.Anaconda.com), which installs    #
#         Python and a bunch of useful scientific computing modules.          #
#                                                                             #
#      2) If you're running Windows, go to the Search bar at the bottom of    #
#         the screen and type in Spyder, then click on Spyder to open it.     #
#         Spyder is an awesome Integrated Development Environment (IDE),      #
#         where you can write and run codes. Run this code from Spyder and    #
#         the plot will appear in the Plots window.                           #
#                                                                             #
#      3) Click the green arrow in the toolbar at the top of the screen to    #
#         run the code. A plot should appear if the code ran successfully.    #
#                                                                             #
#      4) Change the numbers in the [a_,b_,N,R]... line of the code, to       #
#         adjust the line's starting slope and y-intercept (before adding in  #
#         the randomness), number of datapoints, and randomness amplitude,    #
#         respectively.                                                       #
#                                                                             #
#      WARNING: trying to plot the results for N > 10000 may take a very long #
#         time. Recommend keeping N < 10000. You can test the speed of the a  #
#         and b calculations by commenting out the plot lines and running the #
#         code. When running it that way, you will find that the code can     #
#         quickly calculate the line of best fit, even for millions of data   #
#         points!                                                             #
#                                                                             #
#   Questions? Comments? Leave a comment on the Linear Regression video and   #
#   I'll do my best to respond. YouTube @RichBehiel                           #
#                                                                             #
#   Thanks for seeing the beauty in math! :)                                  #
#                                                                             #
#   -Rich                                                                     #
##                                                                            #
###------------------- Beginning of Code: Import Modules -------------------###
import numpy as np; import random; from matplotlib import pyplot as plt       #
###----------------- Make a Dataset & Calculate Best Fit Line --------------###
[a_,b_,N,R] = [2,10,100,6]; x = np.linspace(0,10,N); y = a_*x + b_ # Dataset  #
for i in range(N): y[i] = y[i] + R*(random.random()-0.5) # Add randomness to y#
[SX2,SX,SXY,SY] = [sum(x**2),sum(x),sum(x*y),sum(y)] # Calculate sums of data #
[M,V] = [np.array([[SX2,SX],[SX,N]]),np.array([[SXY],[SY]])] # Matrix & vector#
[a,b] = np.dot(np.linalg.inv(M),V); y_f = a*x + b # Matrix equation; fit line #
###--------------------------- Plot the Results ----------------------------###
plt.subplots(figsize=[8,4.5]); plt.scatter(x,y), plt.plot(x,y_f) # Plot data  #
for i in range(N): plt.plot([x[i],x[i]],[y[i],y_f[i]],':',color='r',zorder=0) #
InfoStr='Fit: y = '+"{:.5f}".format(float(a))+'x + '+"{:.5f}".format(float(b))#
plt.xlabel('x\n'+InfoStr), plt.ylabel('y'),plt.title('Datapoints & Line Fit') #
plt.legend(['${x_i,y_i}$ Data','Line Fit','Residuals']) # Add legend to plot  #
###----------------------------- End of Code -------------------------------###
