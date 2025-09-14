import importlib
import os
import numpy as np


def get_generator(name: str):
    """Return a standardized generator callable with signature:
    gen(n_samples=1500, noise_level=0.15, random_state=42) -> (X, Y)
    """
    name = name.lower()
    if name in {"highly", "highly_nonlinear", "default"}:
        from data_gens.highly_nonlinear import generate_highly_nonlinear_data as _gen
        def _wrap_highly(n_samples=1500, noise_level=0.15, random_state=42):
            return _gen(N=n_samples, noise_level=noise_level, random_state=random_state)
        return _wrap_highly

    if name in {"better_xor", "betterxor", "xor_better"}:
        from data_gens.better_xor_data import generate_better_xor_data as _gen
        def _wrap_betterxor(n_samples=1500, noise_level=0.10, random_state=42):
            return _gen(n_samples=n_samples, n_features=101, noise_level=noise_level, random_state=random_state)
        return _wrap_betterxor

    if name in {"nuclear_xor", "nuclear_friendly", "xor_nuclear"}:
        from data_gens.nuclear_friendly_xor_data import generate_nuclear_friendly_xor_data as _gen
        def _wrap_nuclear(n_samples=2000, noise_level=0.0, random_state=42):
            return _gen(n_samples=n_samples, n_latent=8, n_noise_dims=200,
                        snr_x=6.0, snr_y=6.0, random_state=random_state, make_drift=False)
        return _wrap_nuclear

    if name in {"extreme_nuclear", "nuclear_extreme", "extreme_xor"}:
        from data_gens.nuclear_friendly_xor_data import generate_extreme_nuclear_friendly_data as _gen
        def _wrap_extreme_nuclear(n_samples=2000, noise_level=0.0, random_state=123):
            return _gen(n_samples=n_samples, random_state=random_state)
        return _wrap_extreme_nuclear

    if name in {"extreme1", "extreme_nonlin_1"}:
        from data_gens.extreme_nonlinear_1 import generate_extreme_nonlinear_1 as _gen
        def _wrap_extreme1(n_samples=3000, noise_level=0.0, random_state=42):
            return _gen(N=n_samples, random_state=random_state)
        return _wrap_extreme1

    if name in {"extreme2", "extreme_nonlin_2"}:
        from data_gens.extreme_nonlinear_2 import generate_extreme_nonlinear_2 as _gen
        def _wrap_extreme2(n_samples=2500, noise_level=0.0, random_state=123):
            return _gen(N=n_samples, random_state=random_state)
        return _wrap_extreme2

    if name in {"extreme3", "extreme_nonlin_3"}:
        from data_gens.extreme_nonlinear_3 import generate_extreme_nonlinear_3 as _gen
        def _wrap_extreme3(n_samples=2000, noise_level=0.0, random_state=7):
            return _gen(N=n_samples, random_state=random_state)
        return _wrap_extreme3

    if name in {"piecewise", "piecewise_nonlinear"}:
        from data_gens.piecewise_nonlinear import generate_piecewise_nonlinear as _gen
        def _wrap_piecewise(n_samples=2000, noise_level=0.0, random_state=11):
            return _gen(n_samples=n_samples, random_state=random_state)
        return _wrap_piecewise

    if name in {"swiss", "swiss_nonlinear"}:
        from data_gens.swiss_nonlinear import generate_swiss_nonlinear as _gen
        def _wrap_swiss(n_samples=2500, noise_level=0.0, random_state=21):
            return _gen(n_samples=n_samples, random_state=random_state)
        return _wrap_swiss

    if name in {"multiscale", "multiscale_interaction", "multi_scale"}:
        from data_gens.multiscale_interaction import generate_multiscale_interaction_data as _gen
        def _wrap_multiscale(n_samples=3000, noise_level=0.08, random_state=42):
            return _gen(n_samples=n_samples, noise_level=noise_level, random_state=random_state)
        return _wrap_multiscale

    # Real dataset: kin8nm (lightweight ARFF loader)
    # We keep it here to reuse the existing 'main' pipeline which expects a generator signature.
    if name in {"kin8nm"}:
        def _wrap_kin8nm(n_samples=None, noise_level=0.0, random_state=42):  # noise_level ignored
            """Load the kin8nm regression dataset from local ARFF file.

            Parameters
            ----------
            n_samples : int | None
                Optional subset size. If provided and smaller than full dataset, a deterministic
                subset (first n_samples rows) is returned. (Could be randomized if needed.)
            noise_level : float
                Unused (kept for signature compatibility).
            random_state : int
                Unused currently; placeholder for potential randomized sub-sampling.
            Returns
            -------
            X : (N, D) ndarray
            Y : (N, 1) ndarray target
            """
            # Resolve ARFF path relative to project root (data_gens/../data_real/kin8nm/...)
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            arff_path = os.path.join(base_dir, "data_real", "kin8nm", "dataset_2175_kin8nm.arff")
            if not os.path.exists(arff_path):
                raise FileNotFoundError(f"kin8nm ARFF file not found at {arff_path}")
            data_started = False
            rows = []
            with open(arff_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    low = line.lower()
                    if low.startswith('@data'):
                        data_started = True
                        continue
                    if not data_started:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    # Skip instances with missing values
                    if any(p == '?' or p == '' for p in parts):
                        continue
                    try:
                        vals = [float(p) for p in parts]
                    except ValueError:
                        # Skip malformed line
                        continue
                    rows.append(vals)
            if not rows:
                raise RuntimeError("No data rows parsed from kin8nm ARFF file")
            arr = np.asarray(rows, dtype=float)
            X = arr[:, :-1]
            Y = arr[:, -1:]
            if n_samples is not None and n_samples < X.shape[0]:
                # Deterministic subset (could shuffle with random_state if desired)
                X = X[:n_samples]
                Y = Y[:n_samples]
            return X, Y
        return _wrap_kin8nm

    raise ValueError(f"Unknown dataset generator name: {name}")


