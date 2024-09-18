import matplotlib as mpl
import matplotlib.transforms as transforms
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
# Although Axes3D is not used directly,
# it is imported because it is needed for 3d projection.


# All functions are for visualization.
def make_ellipses(ax, means, covs, edgecolor, m_color, ls='-', n_std=3):
    def _make_ellipses(mean, cov):
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = mpl.patches.Ellipse((0, 0),
                                      ell_radius_x * 2,
                                      ell_radius_y * 2,
                                      facecolor='none',
                                      edgecolor=edgecolor, lw=1, ls=ls)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
        transf = transf.translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        ax.add_artist(ellipse)
        ax.set_aspect('equal', 'datalim')
        ax.scatter(mean[0], mean[1], c=m_color, marker='*')
    for mean, cov in zip(means, covs):
        _make_ellipses(mean, cov)


def make_1dplot(ax, X, means, covs, mix_prob, info, title):
    def _curve(means, covs, mix_prob, color, label, ls='-'):
        X_range = np.linspace(X.min()-1, X.max()+1, 200).reshape(-1, 1)
        c = len(means)
        prob = np.zeros((X_range.shape[0], c))
        for i in range(c):
            prob[:, i] = norm(means[i], covs[i]).pdf(X_range).flatten()
        prob = (prob * mix_prob).sum(axis=1)
        ax.plot(X_range, prob, c=color, linewidth=1.5, label=label, ls=ls)
    ax.set_title(title)
    _curve(means, covs, mix_prob, 'b', label='Real Model', ls=':')
    _curve(info['means'], np.sqrt(info['covs']), info['mix_prob'],
           'tab:red', label='Estimate Model')
    ax.legend(loc='upper right')


def objective_function_plot(ax, training_info, color):
    x = np.arange(len(training_info))
    obj = [info['objective_function'] for info in training_info]
    ax.plot(x, obj, c=color, linewidth=1.5)
    ax.set_xlabel('iteration')
    ax.set_title('Objective Function per Iteration')
