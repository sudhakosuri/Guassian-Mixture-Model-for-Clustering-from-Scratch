import matplotlib.pyplot as plt

import numpy as np

n_size = 50
mu1, sigma1 = 0, 1  # mean and variance
mu2, sigma2 = 5, 1  # mean and variance
mu3, sigma3 = 10, 1  # mean and variance

x1 = np.random.normal(mu1, np.sqrt(sigma1), n_size)
x2 = np.random.normal(mu2, np.sqrt(sigma2), n_size)
x3 = np.random.normal(mu3, np.sqrt(sigma3), n_size)

X = np.array(list(x1) + list(x2) + list(x3))


def pdf(data, mean: float, variance: float, thet: float):
    a = (1 / (np.sqrt(2 * np.pi * variance)))
    b = np.exp(-(np.square(data - mean) / (2 * variance)))
    return a * b




bins = np.linspace(np.min(X), np.max(X), 100)
k = 3
theta = np.ones((k)) / k
means = np.random.choice(X, k)
variances = np.random.random_sample(size=k)

X = np.array(X)

for step in range(20):

    if step % 1 == 0:
        plt.figure(figsize=(10, 6))
        axes = plt.gca()
        plt.xlabel("$X$")
        plt.ylabel("PDF")
        plt.title("Iteration {}".format(step))
        plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

        plt.plot(bins, pdf(bins, mu1, sigma1), color='grey',
                 label="True pdf")
        plt.plot(bins, pdf(bins, mu2, sigma2), color='grey')
        plt.plot(bins, pdf(bins, mu3, sigma3), color='grey')

        plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue',label="Cluster 1")
        plt.plot(bins, pdf(bins, means[1], variances[1]), color='green', label="Cluster 2")
        plt.plot(bins, pdf(bins, means[2], variances[2]), color='magenta', label="Cluster 3")

        plt.legend(loc='upper right')
        plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
        plt.show()

    # calculate the maximum likelihood of each observation xi
    z = []

    # Loop N not required because using np arrays
    # Expectation step
    for j in range(k):
        z.append(pdf(X, means[j], np.sqrt(variances[j]), theta[j]))
    z = np.array(z)

    b = []
    # Maximizing the expected

    for j in range(k):
        # Normalize
        b.append((z[j]) / (np.sum([z[i] for i in range(k)], axis=0)))

        means[j] = np.sum(b[j] * X) / (np.sum(b[j]))
        variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j]))
        theta[j] = np.mean(b[j])

#End of code



#Start of accuracy calculations

estimated_distributions = []

for m in range(len(X)):
    max_arr = []

    max_arr.append(b[0][m])
    max_arr.append(b[1][m])
    max_arr.append(b[2][m])
    max = 0
    if (max_arr[0] > max_arr[1]) and (max_arr[0] > max_arr[2]):
        max = 1
    elif (max_arr[1] > max_arr[0]) and (max_arr[1] > max_arr[2]):
        max = 2
    else:
        max = 3

    estimated_distributions.append(max)

actual_cluster = [0, 0, 0]
min_val = np.min(means)
max_val = np.max(means)

actual_cluster[0] = np.where(means == min_val)[0][0] + 1
actual_cluster[2] = np.where(means == max_val)[0][0] + 1

for i in range(3):
    if (i + 1) != actual_cluster[0] and (i + 1) != actual_cluster[2]:
        actual_cluster[1] = i + 1

accuracy_distribution1 = 0
accuracy_distribution2 = 0
accuracy_distribution3 = 0
for n in range(len(X)):
    if 0 <= n < 50:
        if estimated_distributions[n] == actual_cluster[0]:
            accuracy_distribution1 += 1
    if 50 <= n < 100:
        if estimated_distributions[n] == actual_cluster[1]:
            accuracy_distribution2 += 1
    if 100 <= n < 149:
        if estimated_distributions[n] == actual_cluster[2]:
            accuracy_distribution3 += 1

print("Accuracy of distribution with mean 0 is  " + str((accuracy_distribution1 / 50) * 100))
print("Accuracy of distribution with mean 5 is  " + str((accuracy_distribution2 / 50) * 100))
print("Accuracy of distribution with mean 10 is  " + str((accuracy_distribution3 / 50) * 100))


Mean_Accuracy_1 = abs(mu1 - abs(means[actual_cluster[0] - 1]))
Mean_Accuracy_2 = abs(mu2 - abs(means[actual_cluster[1] - 1]))
Mean_Accuracy_3 = abs(mu3 - abs(means[actual_cluster[2] - 1]))

Variance_Accuracy_1 = abs(sigma1 - variances[actual_cluster[0] - 1])
Variance_Accuracy_2 = abs(sigma2 - variances[actual_cluster[1] - 1])
Variance_Accuracy_3 = abs(sigma3 - variances[actual_cluster[2] - 1])

print("GMM predicts a mean of " + str(means[actual_cluster[0]-1]) +" for distribution with mean 0. So the deviation/error is " + str(Mean_Accuracy_1))
print("GMM predicts a mean of " + str(means[actual_cluster[1]-1]) +" for distribution with mean 5. So the deviation/error is " + str(Mean_Accuracy_2))
print("GMM predicts a mean of " + str(means[actual_cluster[2]-1]) +" for distribution with mean 10. So the deviation/error is " + str(Mean_Accuracy_3))

print("GMM predicts a variance of " + str(variances[actual_cluster[0]-1]) +" for distribution with mean 0 and variance " + str(sigma1) + ". So the deviation/error is " + str(Variance_Accuracy_1))
print("GMM predicts a variance of " + str(variances[actual_cluster[1]-1]) +" for distribution with mean 5 and variance " + str(sigma2) + ". So the deviation/error is " + str(Variance_Accuracy_2))
print("GMM predicts a variance of " + str(variances[actual_cluster[2]-1]) +" for distribution with mean 10 and variance " + str(sigma3) + ". So the deviation/error is " + str(Variance_Accuracy_3))