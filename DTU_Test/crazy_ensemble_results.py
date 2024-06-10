import matplotlib.pyplot as plt
import numpy as np

X = np.arange(1, 9)
MRR = [0.8162, 0.9059, 0.9045, 0.9037, 0.8445, 0.8492, 0.8267, 0.8034]
PR = [0.7828, 0.7568, 0.6024,0.5165, 0.2275, 0.2290, 0.1741, 0.1322]
ROC = [0.9627, 0.9522, 0.8851, 0.8803, 0.7781, 0.7264, 0.6633, 0.5503]

plt.title('Metrics for different ensemble sizes')

plt.plot(X, MRR, label='MRR', color='blue')
plt.plot(X, PR, label='PR AUC', color='red', alpha=0.6)
plt.plot(X, ROC, label='ROC AUC', color='green', alpha=0.6)

# plot max value of each metric
plt.scatter(X[MRR.index(max(MRR))], max(MRR), color='blue', label='Max MRR')
plt.scatter(X[PR.index(max(PR))], max(PR), color='red', label='Max PR AUC')
plt.scatter(X[ROC.index(max(ROC))], max(ROC), color='green', label='Max ROC AUC')

plt.grid()
plt.legend()
plt.xlim(1,8)
plt.ylim(0,1)
plt.xlabel('Number of base models')
plt.ylabel('Metric value')
plt.show()