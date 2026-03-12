"""
V2V Channel Model with LOS/NLOS classification, path loss, and fading.

Replaces the simplified distance-only communication model with a realistic
propagation model inspired by GEMV2 and 3GPP TR 37.885:

  1. Geometric LOS/NLOS classification (ray-sphere intersection)
  2. Log-distance path loss with type-dependent exponents
  3. Log-normal shadow fading (spatially correlated)
  4. Rician (LOS) / Rayleigh (NLOS) small-scale fading
  5. Link quality mapping via sigmoid over SNR

Jamming zone degradation (D_i * D_j) is preserved and applied on top.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class LinkType(Enum):
    LOS = "los"
    NLOS_VEHICLE = "nlos_vehicle"
    NLOS_OBSTACLE = "nlos_obstacle"


@dataclass
class ChannelParams:
    """Tunable channel model parameters."""
    # Transmit power (dBm) -- typical DSRC
    tx_power: float = 23.0
    # Carrier frequency (GHz)
    freq_ghz: float = 5.9
    # Reference distance (m)
    d0: float = 1.0
    # Free-space path loss at d0  (Friis: 20*log10(4*pi*d0*f/c))
    pl0: float = 47.86

    # Path loss exponents per link type
    n_los: float = 2.0
    n_nlosv: float = 2.5
    n_nloso: float = 3.5

    # Additional vehicle obstruction loss (dB) for NLOSv
    vehicle_loss_db: float = 12.0

    # Shadow fading standard deviation (dB) per link type
    sigma_los: float = 3.0
    sigma_nlosv: float = 5.0
    sigma_nloso: float = 7.0
    # Temporal correlation factor for shadow fading (0 = i.i.d., 1 = static)
    shadow_correlation: float = 0.8

    # Rician K-factor (linear) for LOS links
    # K=6 dB -> linear ~3.98
    rician_k_los: float = 3.98
    # NLOS uses Rayleigh (K=0)

    # Noise floor (dBm)
    noise_floor: float = -95.0
    # SNR midpoint for sigmoid quality mapping (dB)
    snr_midpoint: float = 10.0
    # Sigmoid steepness
    snr_steepness: float = 0.25

    # Effective vehicle radius for NLOSv ray intersection (meters)
    vehicle_body_radius: float = 2.0

    # Enable/disable individual model components
    enable_shadow_fading: bool = True
    enable_small_scale_fading: bool = True


@dataclass
class LinkState:
    """Per-link state for temporal correlation."""
    link_type: LinkType = LinkType.LOS
    shadow_fading_db: float = 0.0
    small_scale_fading_db: float = 0.0
    path_loss_db: float = 0.0
    received_power_dbm: float = 0.0
    snr_db: float = 0.0
    quality: float = 1.0


class V2VChannelModel:
    """
    Geometry-aware V2V channel model.

    Usage:
        model = V2VChannelModel()
        quality_matrix = model.compute_quality_matrix(
            positions, obstacles, vehicle_radii
        )
    """

    def __init__(self, params: Optional[ChannelParams] = None):
        self.params = params or ChannelParams()
        self._rng = np.random.default_rng(42)
        # Shadow fading state for temporal correlation: (i, j) -> last_value
        self._shadow_state: dict[tuple[int, int], float] = {}
        # Link state cache for visualization / debugging
        self._link_states: dict[tuple[int, int], LinkState] = {}

    def reset(self):
        """Reset fading state (e.g., on simulation reset)."""
        self._shadow_state.clear()
        self._link_states.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_quality_matrix(
        self,
        positions: np.ndarray,
        obstacle_centers: list[list[float]] = None,
        obstacle_radii: list[float] = None,
    ) -> np.ndarray:
        """
        Compute NxN pairwise communication quality matrix.

        Args:
            positions: (N, 3) array of agent positions.
            obstacle_centers: list of [x,y,z] for each obstacle sphere.
            obstacle_radii: list of radii matching obstacle_centers.

        Returns:
            (N, N) quality matrix with values in [0, 1].
            Diagonal is 1.0 (self-link).
        """
        n = positions.shape[0]
        quality = np.eye(n)

        obs_c = np.array(obstacle_centers) if obstacle_centers else np.empty((0, 3))
        obs_r = np.array(obstacle_radii) if obstacle_radii else np.empty(0)

        for i in range(n):
            for j in range(i + 1, n):
                q = self._compute_link_quality(
                    i, j, positions[i], positions[j],
                    positions, obs_c, obs_r,
                )
                quality[i, j] = q
                quality[j, i] = q

        return quality

    def compute_pairwise_quality(
        self,
        idx_i: int,
        idx_j: int,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        all_positions: np.ndarray,
        obstacle_centers: np.ndarray,
        obstacle_radii: np.ndarray,
    ) -> float:
        """Compute quality for a single link (for use inside controller loops)."""
        return self._compute_link_quality(
            idx_i, idx_j, pos_i, pos_j,
            all_positions, obstacle_centers, obstacle_radii,
        )

    def get_link_states(self) -> dict[tuple[int, int], LinkState]:
        """Return cached link states for visualization."""
        return dict(self._link_states)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _compute_link_quality(
        self,
        idx_i: int, idx_j: int,
        pos_i: np.ndarray, pos_j: np.ndarray,
        all_positions: np.ndarray,
        obs_centers: np.ndarray, obs_radii: np.ndarray,
    ) -> float:
        p = self.params
        d = np.linalg.norm(pos_j - pos_i)
        if d < 0.01:
            return 1.0

        # 1. Classify link
        link_type = self._classify_link(
            pos_i, pos_j, idx_i, idx_j,
            all_positions, obs_centers, obs_radii,
        )

        # 2. Path loss
        pl_db = self._path_loss(d, link_type)

        # 3. Shadow fading
        shadow_db = 0.0
        if p.enable_shadow_fading:
            shadow_db = self._shadow_fading(idx_i, idx_j, link_type)

        # 4. Small-scale fading
        fading_db = 0.0
        if p.enable_small_scale_fading:
            fading_db = self._small_scale_fading(link_type)

        # 5. Link budget
        prx = p.tx_power - pl_db + shadow_db + fading_db
        snr = prx - p.noise_floor

        # 6. Map to quality [0, 1] via sigmoid
        quality = 1.0 / (1.0 + np.exp(-p.snr_steepness * (snr - p.snr_midpoint)))

        # Cache state
        key = (min(idx_i, idx_j), max(idx_i, idx_j))
        self._link_states[key] = LinkState(
            link_type=link_type,
            shadow_fading_db=shadow_db,
            small_scale_fading_db=fading_db,
            path_loss_db=pl_db,
            received_power_dbm=prx,
            snr_db=snr,
            quality=float(quality),
        )

        return float(np.clip(quality, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 1. LOS / NLOS classification
    # ------------------------------------------------------------------

    def _classify_link(
        self,
        pos_i: np.ndarray, pos_j: np.ndarray,
        idx_i: int, idx_j: int,
        all_positions: np.ndarray,
        obs_centers: np.ndarray, obs_radii: np.ndarray,
    ) -> LinkType:
        """
        Determine link type using ray-sphere intersection.

        Checks obstacles first (NLOSo is worse than NLOSv), then vehicles.
        """
        origin = pos_i
        direction = pos_j - pos_i
        ray_len = np.linalg.norm(direction)
        if ray_len < 1e-6:
            return LinkType.LOS
        direction = direction / ray_len

        # Check obstacle occlusion
        if obs_centers.shape[0] > 0:
            for k in range(obs_centers.shape[0]):
                if _ray_intersects_sphere(origin, direction, ray_len,
                                          obs_centers[k], obs_radii[k]):
                    return LinkType.NLOS_OBSTACLE

        # Check vehicle body occlusion
        vr = self.params.vehicle_body_radius
        for k in range(all_positions.shape[0]):
            if k == idx_i or k == idx_j:
                continue
            if _ray_intersects_sphere(origin, direction, ray_len,
                                      all_positions[k], vr):
                return LinkType.NLOS_VEHICLE

        return LinkType.LOS

    # ------------------------------------------------------------------
    # 2. Path loss
    # ------------------------------------------------------------------

    def _path_loss(self, d: float, link_type: LinkType) -> float:
        """Log-distance path loss in dB."""
        p = self.params
        d_eff = max(d, p.d0)

        if link_type == LinkType.LOS:
            n = p.n_los
            extra = 0.0
        elif link_type == LinkType.NLOS_VEHICLE:
            n = p.n_nlosv
            extra = p.vehicle_loss_db
        else:
            n = p.n_nloso
            extra = 0.0

        return p.pl0 + 10.0 * n * np.log10(d_eff / p.d0) + extra

    # ------------------------------------------------------------------
    # 3. Shadow fading (log-normal, temporally correlated)
    # ------------------------------------------------------------------

    def _shadow_fading(self, idx_i: int, idx_j: int, link_type: LinkType) -> float:
        """Correlated log-normal shadow fading in dB."""
        p = self.params
        key = (min(idx_i, idx_j), max(idx_i, idx_j))

        sigma = {
            LinkType.LOS: p.sigma_los,
            LinkType.NLOS_VEHICLE: p.sigma_nlosv,
            LinkType.NLOS_OBSTACLE: p.sigma_nloso,
        }[link_type]

        innovation = float(self._rng.normal(0.0, sigma))

        prev = self._shadow_state.get(key, 0.0)
        corr = p.shadow_correlation
        value = corr * prev + (1.0 - corr) * innovation

        self._shadow_state[key] = value
        return value

    # ------------------------------------------------------------------
    # 4. Small-scale fading (Rician / Rayleigh)
    # ------------------------------------------------------------------

    def _small_scale_fading(self, link_type: LinkType) -> float:
        """
        Small-scale fading gain in dB.

        LOS: Rician envelope with K-factor
        NLOS: Rayleigh envelope (K=0)
        """
        if link_type == LinkType.LOS:
            K = self.params.rician_k_los
            # Rician: envelope = sqrt((x + sqrt(2K))^2 + y^2) / sqrt(2(K+1))
            # where x, y ~ N(0, 1)
            x = float(self._rng.normal())
            y = float(self._rng.normal())
            s = np.sqrt(2.0 * K)
            envelope = np.sqrt((x + s) ** 2 + y ** 2) / np.sqrt(2.0 * (K + 1.0))
        else:
            # Rayleigh: envelope = sqrt(x^2 + y^2), x,y ~ N(0, 1/sqrt(2))
            x = float(self._rng.normal(0, 1.0 / np.sqrt(2.0)))
            y = float(self._rng.normal(0, 1.0 / np.sqrt(2.0)))
            envelope = np.sqrt(x ** 2 + y ** 2)

        # Prevent log(0) and extreme values
        envelope = max(envelope, 1e-6)
        gain_db = 20.0 * np.log10(envelope)
        return float(np.clip(gain_db, -20.0, 10.0))


# ======================================================================
# Geometry utilities
# ======================================================================

def _ray_intersects_sphere(
    origin: np.ndarray,
    direction: np.ndarray,
    ray_length: float,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> bool:
    """
    Test if a ray segment intersects a sphere.

    Uses the geometric solution (fast, no sqrt needed for rejection).
    """
    oc = origin - sphere_center
    b = np.dot(oc, direction)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - c

    if discriminant < 0:
        return False

    sqrt_disc = np.sqrt(discriminant)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc

    # Intersection within ray segment [0, ray_length]?
    return t2 >= 0.0 and t1 <= ray_length


# ======================================================================
# Singleton
# ======================================================================

_channel_model: Optional[V2VChannelModel] = None


def get_channel_model() -> V2VChannelModel:
    """Get or create the global V2V channel model instance."""
    global _channel_model
    if _channel_model is None:
        _channel_model = V2VChannelModel()
    return _channel_model


def reset_channel_model():
    """Reset the global channel model."""
    global _channel_model
    if _channel_model is not None:
        _channel_model.reset()
    _channel_model = None
