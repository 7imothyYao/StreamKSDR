"""Realtime Online KSPCA implementation (numpy-only core)."""

import numpy as np
from .rff import RFF


class OnlineKernelSDR:
    """Online Kernel Supervised Dimensionality Reduction (formerly RealtimeOnlineKSPCA).

    Provides realtime per-sample supervised subspace learning with adaptive / advanced
    learning rate strategies, stability controls, optional whitening, and diagnostic
    introspection. The class name was renamed from RealtimeOnlineKSPCA to
    OnlineKernelSDR for clarity and conciseness. The old name remains as an alias
    for backward compatibility.
    """

    def __init__(self,
                 d_x,
                 d_y,
                 k,
                 D_x,
                 D_y,
                 sigma_x=1.0,
                 sigma_y=1.0,
                 kernel_x='rbf',
                 kernel_y='rbf',
                 base_lr=0.01,
                 adaptive_lr=True,
                 advanced_adaptive=False,
                 warmup_steps=40,
                 target_grad_lower=0.12,
                 target_grad_upper=1.8,
                 increase_factor=1.08,
                 decrease_factor=0.65,
                 recover_factor=1.02,
                 plateau_u_change=2e-3,
                 plateau_boost=1.05,
                 adapt_every=5,
                 lr_decay_factor=0.95,
                 min_lr=1e-5,
                 max_lr=0.1,
                 decay_mode='none',
                 gradient_clip=1.0,
                 step_clip=None,
                 convergence_window=50,
                 stability_threshold=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 forgetting=1.0,
                 whiten=False,
                 max_u_change=None,
                 random_state=12):
        # ---------------- RFF mappings ----------------
        self.rff_x = RFF(d_x, D_x, sigma_x, kernel_x, random_state)
        self.rff_y = RFF(d_y, D_y, sigma_y, kernel_y, random_state + 1)

        # ---------------- Dimensions ----------------
        self.k = k
        self.D_x = D_x
        self.D_y = D_y

        # ---------------- Learning rate (basic + advanced flags) ----------------
        self.base_lr = base_lr
        self.adaptive_lr = adaptive_lr
        self.advanced_adaptive = advanced_adaptive
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay_mode = decay_mode.lower()
        self.current_lr = base_lr

        # ---------------- Advanced adaptive parameters ----------------
        self.warmup_steps = int(warmup_steps)
        self.target_grad_lower = float(target_grad_lower)
        self.target_grad_upper = float(target_grad_upper)
        self.increase_factor = float(increase_factor)
        self.decrease_factor = float(decrease_factor)
        self.recover_factor = float(recover_factor)
        self.plateau_u_change = float(plateau_u_change)
        self.plateau_boost = float(plateau_boost)
        self.adapt_every = max(1, int(adapt_every))
        self._recent_grad = []
        self._recent_u_change = []

        # ---------------- Stability controls ----------------
        self.gradient_clip = gradient_clip
        self.step_clip = step_clip
        self.convergence_window = convergence_window
        self.stability_threshold = stability_threshold

        # ---------------- Adam params ----------------
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Counters
        self.t = 0

        # Running statistics
        self.mean_x = np.zeros(D_x)
        self.mean_y = np.zeros(D_y)
        self.R_xy = np.zeros((D_x, D_y))

        # Optimizer states
        self.m_U = np.zeros((D_x, k))
        self.v_U = np.zeros((D_x, k))

        # U initialization
        rng = np.random.RandomState(random_state)
        U0 = rng.normal(size=(D_x, k))
        self.U, _ = np.linalg.qr(U0)

        # Monitoring lists
        self.gradient_norms = []
        self.U_changes = []
        self.learning_rates = []

        # Anomalies / control counts
        self.nan_count = 0
        self.large_gradient_count = 0
        self.max_u_change_threshold = max_u_change

        if not (0 < forgetting <= 1.0):
            raise ValueError(f"forgetting must be in (0,1]; got {forgetting}")
        self.forgetting = forgetting

        # Whitening (diagonal) statistics
        self.whiten = bool(whiten)
        self.m2_x = np.zeros(D_x, dtype=np.float64)
        self.m2_y = np.zeros(D_y, dtype=np.float64)
        self.var_eps = 1e-6

    # -------------------------------- internal utils --------------------------------
    def _adaptive_learning_rate(self, gradient_norm):
        # Legacy / basic path if advanced disabled
        if not self.adaptive_lr:  # only schedule
            if self.decay_mode != 'none':
                if self.decay_mode == 'sqrt':
                    self.current_lr = self.base_lr / np.sqrt(max(1, self.t))
                elif self.decay_mode == '1t':
                    self.current_lr = self.base_lr / max(1, self.t)
                elif self.decay_mode == 'exp':
                    self.current_lr = self.base_lr * (self.lr_decay_factor ** self.t)
            self.current_lr = float(np.clip(self.current_lr, self.min_lr, self.max_lr))
            return self.current_lr

        if not self.advanced_adaptive:
            # Original heuristic path
            if self.decay_mode == 'sqrt':
                base = self.base_lr / np.sqrt(max(1, self.t))
            elif self.decay_mode == '1t':
                base = self.base_lr / max(1, self.t)
            elif self.decay_mode == 'exp':
                base = self.base_lr * (self.lr_decay_factor ** self.t)
            else:
                base = self.base_lr

            self.current_lr = base
            if gradient_norm > 2.0:
                self.current_lr *= self.lr_decay_factor
            elif gradient_norm < 0.1 and len(self.gradient_norms) > 10:
                if np.mean(self.gradient_norms[-10:]) < 0.05:
                    self.current_lr /= self.lr_decay_factor
            self.current_lr = float(np.clip(self.current_lr, self.min_lr, self.max_lr))
            return self.current_lr

        # ---------------- Advanced adaptive strategy ----------------
        # Warmup: keep constant base lr
        if self.t <= self.warmup_steps:
            self.current_lr = float(np.clip(self.base_lr, self.min_lr, self.max_lr))
            return self.current_lr

        self._recent_grad.append(gradient_norm)
        if len(self._recent_grad) > 100:
            self._recent_grad = self._recent_grad[-100:]

        # Periodic adaptation
        if self.t % self.adapt_every == 0:
            g_mean = float(np.mean(self._recent_grad[-self.adapt_every:]))
            # Outside upper band -> shrink
            if g_mean > self.target_grad_upper:
                self.current_lr *= self.decrease_factor
            # Below lower band -> expand
            elif g_mean < self.target_grad_lower:
                self.current_lr *= self.increase_factor
            else:
                # Gentle recovery toward base
                if self.current_lr < self.base_lr:
                    self.current_lr *= self.recover_factor
                elif self.current_lr > self.base_lr * 1.2:
                    self.current_lr *= 1 / self.recover_factor

            # Plateau detection using U_change window if available
            if self._recent_u_change:
                uc_mean = float(np.mean(self._recent_u_change[-self.adapt_every:]))
                if uc_mean < self.plateau_u_change and self.target_grad_lower < g_mean < self.target_grad_upper:
                    self.current_lr *= self.plateau_boost

        self.current_lr = float(np.clip(self.current_lr, self.min_lr, self.max_lr))
        return self.current_lr

        # baseline schedule
        if self.decay_mode == 'sqrt':
            base = self.base_lr / np.sqrt(max(1, self.t))
        elif self.decay_mode == '1t':
            base = self.base_lr / max(1, self.t)
        elif self.decay_mode == 'exp':
            base = self.base_lr * (self.lr_decay_factor ** self.t)
        else:
            base = self.base_lr

        self.current_lr = base
        if gradient_norm > 2.0:
            self.current_lr *= self.lr_decay_factor
        elif gradient_norm < 0.1 and len(self.gradient_norms) > 10:
            if np.mean(self.gradient_norms[-10:]) < 0.05:
                self.current_lr /= self.lr_decay_factor
        self.current_lr = float(np.clip(self.current_lr, self.min_lr, self.max_lr))
        return self.current_lr

    def _clip_gradient(self, gradient):
        gnorm = np.linalg.norm(gradient)
        if gnorm > self.gradient_clip:
            self.large_gradient_count += 1
            return gradient * (self.gradient_clip / (gnorm + 1e-12))
        return gradient

    def _check_numerical_stability(self, arr, name):
        if np.any(~np.isfinite(arr)):
            self.nan_count += 1
            print(f"Warning: {name} has NaN/Inf at step {self.t}")
            return False
        return True

    # -------------------------------- public helpers --------------------------------
    def get_covariance(self, unbiased=False):
        C = self.R_xy - np.outer(self.mean_x, self.mean_y)
        if unbiased and self.t > 1:
            return (self.t / (self.t - 1.0)) * C
        return C

    @property
    def C_xy(self):
        return self.get_covariance(False)

    # -------------------------------- core update --------------------------------
    def update(self, x, y):
        self.t += 1
        phi = self.rff_x.transform(x)
        psi = self.rff_y.transform(y)
        if not (self._check_numerical_stability(phi, 'phi') and self._check_numerical_stability(psi, 'psi')):
            return {"status": "numerical_error", "step": self.t}

        if self.forgetting < 1.0:
            lam = self.forgetting
            one_minus = 1.0 - lam
            self.mean_x = lam * self.mean_x + one_minus * phi
            self.mean_y = lam * self.mean_y + one_minus * psi
            self.R_xy = lam * self.R_xy + one_minus * np.outer(phi, psi)
            # second moments for whitening
            if self.whiten:
                self.m2_x = lam * self.m2_x + one_minus * (phi ** 2)
                self.m2_y = lam * self.m2_y + one_minus * (psi ** 2)
            unbiased = False
        else:
            eta = 1.0 / self.t
            self.mean_x += eta * (phi - self.mean_x)
            self.mean_y += eta * (psi - self.mean_y)
            self.R_xy += eta * (np.outer(phi, psi) - self.R_xy)
            if self.whiten:
                self.m2_x += eta * ((phi ** 2) - self.m2_x)
                self.m2_y += eta * ((psi ** 2) - self.m2_y)
            unbiased = True

        if self.t > 1:
            C_xy = self.get_covariance(unbiased=unbiased)
        else:
            C_xy = np.zeros_like(self.R_xy)

        # Apply diagonal whitening to covariance for gradient if enabled
        if self.whiten and self.t > 5:  # wait a few steps for statistics
            var_x = np.maximum(self.m2_x - self.mean_x ** 2, self.var_eps)
            var_y = np.maximum(self.m2_y - self.mean_y ** 2, self.var_eps)
            inv_sx = 1.0 / np.sqrt(var_x)
            inv_sy = 1.0 / np.sqrt(var_y)
            # Normalize each side: C_norm[i,j] = C_xy[i,j] / (sqrt(var_x[i]) * sqrt(var_y[j]))
            C_xy_norm = (C_xy * inv_sx[:, None]) * inv_sy[None, :]
        else:
            C_xy_norm = C_xy

        CxyT_U = C_xy_norm.T @ self.U
        G = 2 * C_xy_norm @ CxyT_U
        UTG = self.U.T @ G
        delta = G - self.U @ ((UTG + UTG.T) / 2)
        delta = self._clip_gradient(delta)
        gnorm = float(np.linalg.norm(delta))
        lr = self._adaptive_learning_rate(gnorm)

        # Adam
        self.m_U = self.beta1 * self.m_U + (1 - self.beta1) * delta
        self.v_U = self.beta2 * self.v_U + (1 - self.beta2) * (delta ** 2)
        m_hat = self.m_U / (1 - self.beta1 ** self.t)
        v_hat = self.v_U / (1 - self.beta2 ** self.t)
        step = lr * m_hat / (np.sqrt(v_hat) + self.eps)
        if self.step_clip is not None:
            sn = np.linalg.norm(step)
            if sn > self.step_clip:
                step *= self.step_clip / (sn + 1e-12)

        U_old = self.U.copy()
        self.U += step
        self.U, R = np.linalg.qr(self.U)
        if not self._check_numerical_stability(self.U, 'U'):
            rng = np.random.RandomState(self.t)
            U0 = rng.normal(size=(self.D_x, self.k))
            self.U, _ = np.linalg.qr(U0)
            return {"status": "reset_U", "step": self.t}

        U_change = np.linalg.norm(self.U - U_old)
        if self.max_u_change_threshold is not None and U_change > self.max_u_change_threshold:
            scale = self.max_u_change_threshold / (U_change + 1e-12)
            self.U = U_old + (self.U - U_old) * scale
            self.U, R = np.linalg.qr(self.U)
            U_change = np.linalg.norm(self.U - U_old)

        self.gradient_norms.append(float(gnorm))
        self.U_changes.append(float(U_change))
        self.learning_rates.append(float(lr))
        if self.advanced_adaptive:
            self._recent_u_change.append(float(U_change))
            if len(self._recent_u_change) > 100:
                self._recent_u_change = self._recent_u_change[-100:]
        if len(self.gradient_norms) > self.convergence_window * 2:
            self.gradient_norms = self.gradient_norms[-self.convergence_window:]
            self.U_changes = self.U_changes[-self.convergence_window:]
            self.learning_rates = self.learning_rates[-self.convergence_window:]

        return {
            "status": "success",
            "step": self.t,
            "gradient_norm": gnorm,
            "U_change": U_change,
            "learning_rate": lr,
            "condition_number": float(np.linalg.cond(R)) if R.size else 1.0
        }

    # -------------------------------- monitoring --------------------------------
    def is_converged(self):
        if len(self.U_changes) < self.convergence_window:
            return False, {"reason": "insufficient_samples", "window_size": len(self.U_changes)}
        recent_changes = self.U_changes[-self.convergence_window:]
        recent_grad = self.gradient_norms[-self.convergence_window:]
        avg_change = float(np.mean(recent_changes))
        std_change = float(np.std(recent_changes))
        avg_grad = float(np.mean(recent_grad))
        converged = (avg_change < self.stability_threshold and
                     std_change < self.stability_threshold * 0.5 and
                     avg_grad < self.stability_threshold * 2)
        info = {
            "avg_change": avg_change,
            "std_change": std_change,
            "avg_gradient": avg_grad,
            "threshold": self.stability_threshold
        }
        return bool(converged), info

    def fit_stream(self, data_stream,
                   max_samples=None,
                   convergence_check_freq=100,
                   verbose=True):
        sample_count = 0
        for x, y in data_stream:
            upd = self.update(x, y)
            sample_count += 1
            if upd["status"] != "success" and verbose:
                print(f"Step {self.t}: {upd['status']}")
            if verbose and sample_count % convergence_check_freq == 0:
                conv, info = self.is_converged()
                print(f"Sample {sample_count}: converged={conv} lr={self.current_lr:.5f} grad={upd.get('gradient_norm',0):.4f} avg_dU={info.get('avg_change',0):.4e}")
                if conv:
                    break
            if max_samples and sample_count >= max_samples:
                if verbose:
                    print(f"Reached max_samples={max_samples}")
                break
        conv_final, info_final = self.is_converged()
        return {
            "total_samples": sample_count,
            "converged": conv_final,
            "convergence_info": info_final,
            "final_lr": self.current_lr,
            "nan_count": self.nan_count,
            "large_gradient_count": self.large_gradient_count,
            "gradient_history": self.gradient_norms.copy(),
            "U_change_history": self.U_changes.copy(),
            "lr_history": self.learning_rates.copy()
        }

    def get_diagnostics(self):
        return {
            "steps": self.t,
            "current_lr": self.current_lr,
            "U_condition": float(np.linalg.cond(self.U)),
            "covariance_norm": float(np.linalg.norm(self.get_covariance())),
            "mean_x_norm": float(np.linalg.norm(self.mean_x)),
            "mean_y_norm": float(np.linalg.norm(self.mean_y)),
            "gradient_stats": {
                "mean": float(np.mean(self.gradient_norms)) if self.gradient_norms else 0.0,
                "std": float(np.std(self.gradient_norms)) if self.gradient_norms else 0.0,
                "last": float(self.gradient_norms[-1]) if self.gradient_norms else 0.0
            },
            "anomaly_counts": {
                "nan_count": self.nan_count,
                "large_gradient_count": self.large_gradient_count
            }
        }

# Backward compatibility alias (deprecated): keep after class definition
RealtimeOnlineKSPCA = OnlineKernelSDR
