import logging
from typing import Literal, Tuple

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RejectionSampler:
    """Generic implementation of the rejection sampling algorithm."""

    def __init__(
        self,
        data: np.ndarray,
        kde_bandwidth: float = None,
        proposal_distribution_params: Tuple[np.ndarray, np.ndarray] = None,
        scaling_mode: Literal["max", "3rd_quartile"] = "max",
    ):
        """Initializes the inner distributions used by the rejection sampling algorithm.

        Args:
            data: N x D array where N is the number of data points and D is the dimensionality of the data.
            kde_bandwidth: Bandwidth of the kernel density estimator. If no bandwidth is given, it will be determined
                by cross-validation over ``data``.
            proposal_distribution_params: `mean` and `cov` parameters to use for the Gaussian proposal distribution. If
                no params are given, the proposal distribution is inferred from the mean and covariance computed on
                ``data``.
            scaling_mode: Algorithm to use to compute the scaling factor between the proposal distribution and the KDE
                estimation of the real distribution.
        """
        self.data = data

        # Init kernel density estimate
        if kde_bandwidth:
            self.kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian").fit(self.data)
        else:
            logger.info("Cross-validating bandwidth of kernel density estimate...")
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"), {"bandwidth": 10 ** np.linspace(-1, 1, 100)}, cv=ShuffleSplit()
            )
            grid.fit(self.data)
            self.kde = grid.best_estimator_

        # Init proposal distribution
        if proposal_distribution_params:
            mean, cov = proposal_distribution_params
        else:
            mean = np.mean(self.data, axis=0)
            cov = np.cov(self.data, rowvar=False)
        self.proposal_distribution = multivariate_normal(mean=mean, cov=cov)

        # Init scaling factor
        factors_between_data_and_proposal_distribution = (
            np.e ** self.kde.score_samples(self.data)
        ) / self.proposal_distribution.pdf(self.data)
        if scaling_mode == "max":
            # 'max' scaling mode is used when the initial samples fit in a sensible distribution
            # Should be preferred to other algorithms whenever it is applicable
            self.scaling_factor = np.max(factors_between_data_and_proposal_distribution)
        else:  # scaling_mode == '3rd_quartile'
            # '3rd_quartile' scaling mode is used when outliers in the initial samples skew the ratio and cause an
            # impossibly high scaling factor
            self.scaling_factor = np.percentile(factors_between_data_and_proposal_distribution, 75)

    def sample(self, n_samples: int) -> np.ndarray:
        """Performs rejection sampling to sample N samples that fit the visible distribution of ``data``.

        Args:
            n_samples: Number of samples to sample from the data distribution.

        Returns:
            M x D array where M equals `nb_samples` and D is the dimensionality of the sampled data.
        """
        samples = np.empty([n_samples, self.data.shape[1]])
        nb_trials = 0
        accepted_samples = 0

        tqdm.write("Press ctrl-c to stop the sampling and continue the pipeline.")
        pbar = tqdm(
            total=n_samples, desc="Sampling from observed data distribution with rejection sampling", unit="sample"
        )
        try:
            while accepted_samples < n_samples:
                sample = self.proposal_distribution.rvs(size=1)
                rand_likelihood_threshold = np.random.uniform(
                    0, self.scaling_factor * self.proposal_distribution.pdf(sample)
                )

                if rand_likelihood_threshold <= (np.e ** self.kde.score_samples(sample[np.newaxis, :])):
                    samples[accepted_samples] = sample
                    accepted_samples += 1
                    pbar.update()

                nb_trials += 1

        except KeyboardInterrupt:
            tqdm.write("Sampling cancelled!")
            samples.resize([accepted_samples, self.data.shape[1]])

        pbar.close()
        logger.info(
            "Percentage of generated samples accepted by rejection sampling: "
            f"{round(samples.shape[0] / nb_trials * 100, 2)} \n"
        )

        return samples
