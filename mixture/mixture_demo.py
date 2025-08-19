import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from custom_gmm import CustomGMM
from copula_mm import CopulaMM
from dgmm import DGMM
from poisson_mm import PoissonMixtureModel
from student_mm import StudentMixtureModel
from gamma_mm import GammaMixtureModel
from dpm_ef import DPMEF

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    num_cores = multiprocessing.cpu_count()

    # Generate synthetic data for GMM, SMM, DPM-EF
    X_mixed = np.vstack([
        np.random.normal(loc=0, scale=1, size=(5000, 10)),
        np.random.poisson(lam=3, size=(5000, 10)),
        np.random.gamma(shape=2, scale=2, size=(5000, 10))
    ])

    # Generate synthetic data for PMM
    X_poisson = np.vstack([
        np.random.poisson(lam=3, size=(5000, 10)),
        np.random.poisson(lam=10, size=(5000, 10))
    ])

    # Generate synthetic data for GammaMM
    X_gamma = np.vstack([
        np.random.gamma(shape=2, scale=3, size=(5000, 10)),
        np.random.gamma(shape=5, scale=1, size=(5000, 10))
    ])

    # Generate synthetic data for CopulaMM, NFMM, DGMM
    X_rand = np.random.rand(10000, 20)

    # Demo CustomGMM
    print("=== CustomGMM Demo ===")
    def train_gmm(n):
        model = CustomGMM(n_components=n)
        model.fit(X_mixed)
        return model
    gmm_models = Parallel(n_jobs=num_cores)(delayed(train_gmm)(n) for n in range(2, 5))
    best_gmm = max(gmm_models, key=lambda m: m.compute_log_likelihood(X_mixed))
    labels_gmm = best_gmm.predict(X_mixed)
    print(f"Best GMM Log-Likelihood: {best_gmm.compute_log_likelihood(X_mixed)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_gmm))}")

    # Demo CopulaMM
    print("\n=== CopulaMM Demo ===")
    def train_copula(n):
        model = CopulaMM(n_components=n)
        model.fit(X_rand)
        return model
    copula_models = Parallel(n_jobs=num_cores)(delayed(train_copula)(n) for n in range(2, 5))
    best_copula = min(copula_models, key=lambda m: m.compute_bic(X_rand))
    labels_copula = best_copula.predict(X_rand)
    print(f"Best CopulaMM BIC: {best_copula.compute_bic(X_rand)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_copula))}")

    # Demo DGMM
    print("\n=== DGMM Demo ===")
    def train_dgmm(n):
        model = DGMM(n_components=n, input_dim=X_rand.shape[1])
        model.fit(X_rand)
        return model
    dgmm_models = Parallel(n_jobs=num_cores)(delayed(train_dgmm)(n) for n in range(2, 5))
    best_dgmm = min(dgmm_models, key=lambda m: m.compute_bic(X_rand))
    labels_dgmm = best_dgmm.predict(X_rand)
    print(f"Best DGMM BIC: {best_dgmm.compute_bic(X_rand)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_dgmm))}")

    # Demo PoissonMixtureModel
    print("\n=== PoissonMixtureModel Demo ===")
    def train_pmm(n):
        model = PoissonMixtureModel(n_components=n)
        model.fit(X_poisson)
        return model
    pmm_models = Parallel(n_jobs=num_cores)(delayed(train_pmm)(n) for n in range(2, 5))
    best_pmm = max(pmm_models, key=lambda m: m.compute_log_likelihood(X_poisson))
    labels_pmm = best_pmm.predict(X_poisson)
    print(f"Best PMM Log-Likelihood: {best_pmm.compute_log_likelihood(X_poisson)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_pmm))}")

    # Demo StudentMixtureModel
    print("\n=== StudentMixtureModel Demo ===")
    def train_smm(n):
        model = StudentMixtureModel(n_components=n)
        model.fit(X_mixed)
        return model
    smm_models = Parallel(n_jobs=num_cores)(delayed(train_smm)(n) for n in range(2, 5))
    best_smm = max(smm_models, key=lambda m: m.compute_log_likelihood(X_mixed))
    labels_smm = best_smm.predict(X_mixed)
    print(f"Best SMM Log-Likelihood: {best_smm.compute_log_likelihood(X_mixed)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_smm))}")

    # Demo GammaMixtureModel
    print("\n=== GammaMixtureModel Demo ===")
    def train_gammamm(n):
        model = GammaMixtureModel(n_components=n)
        model.fit(X_gamma)
        return model
    gammamm_models = Parallel(n_jobs=num_cores)(delayed(train_gammamm)(n) for n in range(2, 5))
    best_gammamm = max(gammamm_models, key=lambda m: m.compute_log_likelihood(X_gamma))
    labels_gammamm = best_gammamm.predict(X_gamma)
    print(f"Best GammaMM Log-Likelihood: {best_gammamm.compute_log_likelihood(X_gamma)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_gammamm))}")

    # Demo DPMEF
    print("\n=== DPMEF Demo ===")
    def train_dpmef(n):
        model = DPMEF(num_clusters=n)
        model.fit(X_mixed)
        return model
    dpmef_models = Parallel(n_jobs=num_cores)(delayed(train_dpmef)(n) for n in range(5, 8))
    best_dpmef = max(dpmef_models, key=lambda m: m.compute_log_likelihood(X_mixed))
    labels_dpmef = best_dpmef.predict(X_mixed)
    print(f"Best DPM-EF Log-Likelihood: {best_dpmef.compute_log_likelihood(X_mixed)}")
    print(f"Number of Unique Clusters: {len(np.unique(labels_dpmef))}")

if __name__ == "__main__":
    main()
