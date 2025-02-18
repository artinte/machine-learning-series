import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixture:
    def __init__(self, mus, covs, weights):
        '''
        mus: (means) a list of K 1d np arrays (D,)
        covs: (covariances) a list of K 2d np arrays (D, D)
        weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each brach.
                They will be normalized to sum to 1. If they sum to zero, it will err.
        '''
        self.n_component = len(mus)
        self.mus = mus
        self.covs = covs    # covariance
        self.precs = [np.linalg.inv(cov) for cov in covs]   # precision
        self.weights = np.array(weights)
        self.norm_weights = self.weights / self.weights.sum()
        self.RVs = []
        for i in range(len(mus)):
            self.RVs.append(multivariate_normal(mus[i], covs[i]))
        self.dim = len(mus[0])

    def add_component(self, mu, cov, weight=1):
        self.mus.append(mu)
        self.covs.append(cov)
        self.precs.append(np.linalg.inv(cov))
        self.RVs.append(multivariate_normal(mu, cov))
        self.weights.append(weight)
        self.norm_weights = self.weights / self.weights.sum()
        self.n_component += 1

    def pdf(self, x):
        '''
        Probability density (PDF) at $x$.
        '''
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        prob = np.dot(component_pdf, self.norm_weights)
        return prob

    def sample(self, N):
        '''
        Draw N samples from Gaussian mixture
        Procedure:
            Draw N samples from each Gaussian
            Draw N indices, according to the weights.
            Choose sample between the branches according to the indices.
        '''
        rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
        all_samples = np.array([rv.rvs(N) for rv in self.RVs])
        gmm_samps = all_samples[rand_component, np.arange(N), :]
        return gmm_samps, rand_component, all_samples
    
    def score(self, x):
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        scores = np.zeros_like(x)
        # 计算每个分量对 score 的贡献
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ self.precs[i]
            scores += participance[:, i:i+1] * gradvec
        return scores

    def score_decompose(self, x):
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)
        
        gradvec_list = []
        for i in range(self.n_component):
            gradvec = -(x - self.mus[i]) @ self.precs[i]
            gradvec_list.append(gradvec)

        return gradvec_list, participance



# 表示一个标准的二维正态分布（独立且方差为 1）。
mu1 = np.array([0.0, 1.0])
Cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
# 表示一个二维正态分布，其中各维度之间有相关性。
mu2 = np.array([2.0, -1.0])
Cov2 = np.array([[2.0, 0.5], [0.5, 1.0]])

RV1 = multivariate_normal(mu1, Cov1)
RV2 = multivariate_normal(mu2, Cov2)

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z1 = RV1.pdf(pos)
Z2 = RV2.pdf(pos)
plt.contour(X, Y, Z1, levels=10, cmap='Blues', alpha=0.5)
plt.contour(X, Y, Z2, levels=10, cmap='Reds', alpha=0.5)
plt.plot(mu1[0], mu1[1], 'bo', label='Mean of RV1')
plt.plot(mu2[0], mu2[1], 'ro', label='Mean of RV2')
plt.legend()
plt.grid()
plt.show()

gmm = GaussianMixture([mu1, mu2], [Cov1, Cov2], [1.0, 1.0])

gmm_samps, rand_component, component_samples = gmm.sample(5000)
print('gmm_samps:', gmm_samps.shape)
print('rand_component:', rand_component.shape)
print('all_samples:', component_samples.shape)

scorevecs = gmm.score(gmm_samps)
print("scorevecs ", scorevecs.shape)

def kdeplot(pnts, label="", ax=None, titlestr=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(x=pnts[:,0], y=pnts[:,1], ax=ax, label=label, **kwargs)
    if titlestr is not None:
        ax.set_title(titlestr)

def quiver_plot(pnts, vecs, *args, **kwargs):
    plt.quiver(pnts[:, 0], pnts[:,1], vecs[:, 0], vecs[:, 1], *args, **kwargs)

figh, ax = plt.subplots(1,1,figsize=[6, 6])
kdeplot(component_samples[0,:,:], label="comp1", )
kdeplot(component_samples[1,:,:], label="comp2", )
plt.title("Empirical density of each component")
plt.show()

kdeplot(gmm_samps, )
plt.title("Empirical density of Gaussian mixture density")
plt.show()

quiver_plot(gmm_samps, scorevecs)
plt.title("Score vector field")
plt.axis("image")
plt.show()

gmm_samps_few, _, _ = gmm.sample(200)
scorevecs_few = gmm.score(gmm_samps_few)
gradvec_list, participance = gmm.score_decompose(gmm_samps_few)
quiver_plot(gmm_samps_few, gradvec_list[0], color="blue", alpha=0.4, scale=45, label="score of gauss mode1")
quiver_plot(gmm_samps_few, gradvec_list[1], color="orange", alpha=0.4, scale=45, label="score of gauss mode2")
plt.legend()
plt.show()

quiver_plot(gmm_samps_few, gradvec_list[0]*participance[:,0:1], color="blue", alpha=0.4, scale=15, label="weighted score of gauss mode1")
quiver_plot(gmm_samps_few, gradvec_list[1]*participance[:,1:2], color="orange", alpha=0.4, scale=15, label="weighted score of gauss mode2")
quiver_plot(gmm_samps_few, scorevecs_few, scale=15, alpha=0.7, width=0.003, label="score of GMM")
plt.legend()
plt.show()

