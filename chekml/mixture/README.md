# Mixture Models Collection

## 1.GammaMixtureModel

### Algorithm Explanation
The **Gamma Mixture Model (GammaMM)** is a probabilistic model that assumes the data is generated from a mixture of Gamma distributions. It uses the **Expectation-Maximization (EM)** algorithm to estimate parameters:
- **E-step**: Compute responsibilities (posterior probabilities) of each data point.  
- **M-step**: Update mixture weights, shape, and scale parameters.  

This model is well-suited for **positive, skewed data** such as waiting times or sizes. Convergence is monitored via log-likelihood changes.

### Parameters
| Parameter     | Type  | Default | Description |
|---------------|-------|---------|-------------|
| `n_components` | int   | -       | Number of Gamma mixture components (clusters). |
| `max_iter`     | int   | 100     | Maximum EM iterations. |
| `tol`          | float | 1e-4    | Convergence tolerance (absolute change in log-likelihood). |

### Demonstration
```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.gamma_mm import GammaMixtureModel

np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Synthetic Gamma data
X_gamma = np.vstack([
    np.random.gamma(shape=2, scale=3, size=(5000, 10)),
    np.random.gamma(shape=5, scale=1, size=(5000, 10))
])

print("=== GammaMixtureModel Demo ===")

def train_gammamm(n):
    model = GammaMixtureModel(n_components=n)
    model.fit(X_gamma)
    return model

gammamm_models = Parallel(n_jobs=num_cores)(delayed(train_gammamm)(n) for n in range(2, 5))
best_gammamm = max(gammamm_models, key=lambda m: m.compute_log_likelihood(X_gamma))

labels = best_gammamm.predict(X_gamma)
print(f"Best Log-Likelihood: {best_gammamm.compute_log_likelihood(X_gamma)}")
print(f"Clusters Found: {len(np.unique(labels))}")
```

## 2.CopulaMM

### Algorithm Explanation
The Copula-based Mixture Model (CopulaMM) combines copulas with a Gaussian Mixture Model (GMM) to handle dependencies between variables that may not be normally distributed. It transforms the data to uniform margins using scaling and quantile transformation, estimates an empirical correlation matrix, generates copula samples via multivariate normal and CDF transformation, and fits a GMM on the transformed data. This allows modeling complex dependencies while assuming Gaussianity in the copula space. Model selection can use BIC.

### Parameters

| Parameter      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **n_components** | Number of mixture components in the underlying GMM.                        |
| **n_init**       | Number of initializations for GMM (default `10`). Higher values improve robustness but increase computation time. |
| **random_state** | Random seed for reproducibility in transformations and GMM fitting (default `42`). |

### Demonstration
Similar to `mixture_demo.py`, here's how to use CopulaMM: generate synthetic random data, train models with varying components in parallel, select the best based on BIC (lower is better), predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.copula_mm import CopulaMM

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for CopulaMM
X_rand = np.random.rand(10000, 20)

print("=== CopulaMM Demo ===")
def train_copula(n):
    model = CopulaMM(n_components=n)
    model.fit(X_rand)
    return model

copula_models = Parallel(n_jobs=num_cores)(delayed(train_copula)(n) for n in range(2, 5))
best_copula = min(copula_models, key=lambda m: m.compute_bic(X_rand))
labels_copula = best_copula.predict(X_rand)

print(f"Best CopulaMM BIC: {best_copula.compute_bic(X_rand)}")
print(f"Number of Unique Clusters: {len(np.unique(labels_copula))}")
```

## 3.PoissonMixtureModel

### Algorithm Explanation
The Poisson Mixture Model (PMM) models count data as a mixture of Poisson distributions using the Expectation-Maximization (EM) algorithm.  
- **E-step:** responsibilities are calculated based on the Poisson PMF.  
- **M-step:** weights and lambda parameters (means) are updated.  
It's ideal for non-negative integer data like event counts. Convergence is checked using changes in log-likelihood.

### Parameters

| Parameter        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **n_components** | Number of mixture components (Poisson distributions).                       |
| **max_iter**     | Maximum number of EM iterations (default `100`).                            |
| **tol**          | Convergence tolerance for log-likelihood (default `1e-4`).                  |

### Demonstration
Similar to `mixture_demo.py`, here's how to use PoissonMixtureModel: generate synthetic Poisson data, train models with varying components in parallel, select the best based on log-likelihood, predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.poisson_mm import PoissonMixtureModel

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for PMM
X_poisson = np.vstack([
    np.random.poisson(lam=3, size=(5000, 10)),
    np.random.poisson(lam=10, size=(5000, 10))
])

print("=== PoissonMixtureModel Demo ===")
def train_pmm(n):
    model = PoissonMixtureModel(n_components=n)
    model.fit(X_poisson)
    return model

pmm_models = Parallel(n_jobs=num_cores)(delayed(train_pmm)(n) for n in range(2, 5))
best_pmm = max(pmm_models, key=lambda m: m.compute_log_likelihood(X_poisson))
labels_pmm = best_pmm.predict(X_poisson)

print(f"Best PMM Log-Likelihood: {best_pmm.compute_log_likelihood(X_poisson)}")
print(f"Number of Unique Clusters: {len(np.unique(labels_pmm))}")
```

## 4.CustomGMM

### Algorithm Explanation
The Custom Gaussian Mixture Model (CustomGMM) is a from-scratch implementation of GMM using the Expectation-Maximization (EM) algorithm.  
- **E-step:** computes responsibilities via Gaussian PDFs.  
- **M-step:** updates mixture weights, means, and covariance matrices.  
The data is standardized for numerical stability. This model is suitable for continuous data clustering.

### Parameters

| Parameter        | Description                                                                |
|------------------|----------------------------------------------------------------------------|
| **n_components** | Number of Gaussian components.                                             |
| **max_iter**     | Maximum number of EM iterations (default `100`).                           |
| **tol**          | Convergence tolerance for log-likelihood (default `1e-4`).                 |

### Demonstration
Similar to `mixture_demo.py`, here's how to use CustomGMM: generate synthetic mixed data, train models with varying components in parallel, select the best based on log-likelihood, predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.custom_gmm import CustomGMM

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for GMM
X_mixed = np.vstack([
    np.random.normal(loc=0, scale=1, size=(5000, 10)),
    np.random.poisson(lam=3, size=(5000, 10)),
    np.random.gamma(shape=2, scale=2, size=(5000, 10))
])

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
```

## 5.DGMM

### Algorithm Explanation
The **Deep Gaussian Mixture Model (DGMM)** combines an autoencoder with a Gaussian Mixture Model (GMM).  
- The **autoencoder** reduces dimensionality by learning a compact latent representation.  
- The **GMM** then clusters data in this encoded feature space.  
This makes DGMM particularly useful for **high-dimensional data** where traditional GMMs may struggle.  
Model selection is typically performed using **BIC**.

### Parameters

| Parameter        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **n_components** | Number of GMM components.                                                   |
| **input_dim**    | Dimensionality of the input data (number of features).                      |
| **encoding_dim** | Dimensionality of the encoded latent space (default `10`). Lower values compress more. |
| **n_init**       | Number of GMM initializations (default `10`).                               |
| **random_state** | Random seed for reproducibility (default `42`).                             |

### Demonstration
Similar to `mixture_demo.py`, here's how to use DGMM: generate synthetic data, train models with varying components in parallel, select the best based on BIC, predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.dgmm import DGMM

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for DGMM
X_rand = np.random.rand(10000, 20)

print("=== DGMM Demo ===")
def train_dgmm(n):
    model = DGMM(n_components=n, input_dim=X_rand.shape[1])
    model.fit(X_rand, epochs=50, batch_size=256)
    return model

dgmm_models = Parallel(n_jobs=num_cores)(delayed(train_dgmm)(n) for n in range(2, 5))
best_dgmm = min(dgmm_models, key=lambda m: m.compute_bic(X_rand))
labels_dgmm = best_dgmm.predict(X_rand)

print(f"Best DGMM BIC: {best_dgmm.compute_bic(X_rand)}")
print(f"Number of Unique Clusters: {len(np.unique(labels_dgmm))}")
```

## 6.StudentMixtureModel

### Algorithm Explanation
The **Student-t Mixture Model (SMM)** represents data as a mixture of multivariate Student-t distributions.  
Unlike Gaussian mixtures, the **Student-t distribution has heavier tails**, making it more **robust to outliers**.  
- **E-step:** Compute responsibilities using Student-t PDFs.  
- **M-step:** Update mixture weights, means, covariances, and degrees of freedom (`df`).  
The degrees of freedom are iteratively optimized during training.  
This model is well-suited for **data containing outliers** or **non-Gaussian noise**.

### Parameters

| Parameter        | Description                                                                |
|------------------|----------------------------------------------------------------------------|
| **n_components** | Number of Student-t mixture components.                                    |
| **max_iter**     | Maximum number of EM iterations (default `100`).                           |
| **tol**          | Convergence tolerance for log-likelihood change (default `1e-4`).          |

### Demonstration
Similar to `mixture_demo.py`, here's how to use SMM: generate synthetic data, train models with varying components in parallel, select the best based on log-likelihood, predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.student_mm import StudentMixtureModel

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for SMM
X_mixed = np.vstack([
    np.random.normal(loc=0, scale=1, size=(5000, 10)),
    np.random.poisson(lam=3, size=(5000, 10)),
    np.random.gamma(shape=2, scale=2, size=(5000, 10))
])

print("=== StudentMixtureModel Demo ===")
def train_smm(n):
    model = StudentMixtureModel(n_components=n)
    model.fit(X_mixed)
    return model

smm_models = Parallel(n_jobs=num_cores)(delayed(train_smm)(n) for n in range(2, 5))
best_smm = max(smm_models, key=lambda m: m.compute_log_likelihood(X_mixed))
labels_smm = best_smm.predict(X_mixed)

print(f"Best SMM Log-Likelihood: {best_smm.compute_log_likelihood(X_mixed)}")
print(f"Number of Unique Clusters: {len(np.unique(labels_smm))}")
```

## 7.Normalizing Flow Mixture Model (NFMM) (Tensorflow >= 2.18

### Algorithm Explanation
The **Normalizing Flow Mixture Model (NFMM)** combines **normalizing flows** (e.g., RealNVP) with a Gaussian Mixture Model (GMM).  
Normalizing flows transform complex data distributions into a simple **base Gaussian distribution** using invertible transformations.  
After transformation, a GMM is fitted in the latent space.  
This allows NFMM to model **highly non-Gaussian and complex densities** while retaining tractable likelihoods.  
Model selection typically uses **Bayesian Information Criterion (BIC)**.

### Parameters

| Parameter        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **n_components** | Number of Gaussian mixture components.                                      |
| **event_dims**   | Dimensionality of the input data (number of features).                      |
| **num_flows**    | Number of flow transformations in the RealNVP chain (default `5`).          |
| **n_init**       | Number of GMM initializations for robustness (default `10`).                |
| **random_state** | Random seed for reproducibility (default `42`).                             |

### Demonstration
Similar to `mixture_demo.py`, here's how to use NFMM: generate synthetic data, train models with varying components in parallel, select the best based on BIC, predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.nfmm import NFMM

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for NFMM
X_rand = np.random.rand(10000, 20)

print("=== NFMM Demo ===")
def train_nfmm(n):
    model = NFMM(n_components=n, event_dims=X_rand.shape[1])
    model.fit(X_rand)
    return model

nfmm_models = Parallel(n_jobs=num_cores)(delayed(train_nfmm)(n) for n in range(2, 5))
best_nfmm = min(nfmm_models, key=lambda m: m.compute_bic(X_rand))
labels_nfmm = best_nfmm.predict(X_rand)

print(f"Best NFMM BIC: {best_nfmm.compute_bic(X_rand)}")
print(f"Number of Unique Clusters: {len(np.unique(labels_nfmm))}")
```

## 8.Dirichlet Process Mixture of Exponential Families (DPM-EF)

### Algorithm Explanation
The **Dirichlet Process Mixture of Exponential Families (DPM-EF)** is a **non-parametric Bayesian model** that allows for an **infinite number of mixture components**.  
It is typically approximated via **stick-breaking construction**, where mixture weights are drawn sequentially until they approximately sum to one.  
Each component belongs to the **exponential family of distributions** (here, a Normal distribution for demonstration).  
The model automatically adapts to the complexity of the data by inferring the effective number of clusters, without requiring a fixed `n_components`.  
Responsibilities are computed using posterior probabilities, and cluster assignments emerge naturally.  

### Parameters

| Parameter        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **num_clusters** | Truncation level to approximate the infinite mixture (default `10`).        |
| **alpha**        | Dirichlet Process concentration parameter (default `1.0`). Higher values lead to more clusters. |

### Demonstration
Similar to `mixture_demo.py`, here's how to use DPM-EF: generate synthetic data, train models with varying truncation levels in parallel, select the best based on log-likelihood, predict labels, and print results.

```python
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from chekml.mixture.dpm_ef import DPMEF

# Set random seed for reproducibility
np.random.seed(42)
num_cores = multiprocessing.cpu_count()

# Generate synthetic data for DPM-EF
X_mixed = np.vstack([
    np.random.normal(loc=0, scale=1, size=(5000, 10)),
    np.random.poisson(lam=3, size=(5000, 10)),
    np.random.gamma(shape=2, scale=2, size=(5000, 10))
])

print("=== DPMEF Demo ===")
def train_dpmef(n):
    model = DPMEF(num_clusters=n)
    model.fit(X_mixed)
    return model

dpmef_models = Parallel(n_jobs=num_cores)(delayed(train_dpmef)(n) for n in range(5, 8))
best_dpmef = max(dpmef_models, key=lambda m: m.compute_log_likelihood(X_mixed))
labels_dpmef = best_dpmef.predict(X_mixed)

print(f"Best DPM-EF Log-Likelihood: {best_dpmef.compute_log_likelihood(X_mixed)}")
print(f"Number of Unique Clusters: {len(np.unique(labels_dpmef))}")
```
