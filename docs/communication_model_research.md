# Multi-Vehicle Simulation Communication Model Research

## 1. Problem Statement

The current simulation uses a simplified communication model based solely on inter-agent distance:

- **Far-field quality** `a_ij = exp(-alpha * (2^delta - 1) * (r_ij / r0)^v)` -- exponential path-loss decay
- **Near-field quality** `g_ij = r_ij / sqrt(r_ij^2 + r0^2)` -- sigmoidal coupling
- **Pairwise quality** `phi_ij = g_ij * a_ij`

This model has no concept of:

- Line-of-sight (LOS) vs non-line-of-sight (NLOS) propagation
- Vehicle/building obstruction causing shadow regions
- Multipath fading (Rician for LOS, Rayleigh for NLOS)
- Log-normal shadow fading
- Realistic link budget (SNR, receiver sensitivity)

The result is that all agents at the same distance have identical link quality, regardless of obstacles between them. This is unrealistic for a vehicle swarm scenario.

---

## 2. Literature Review

### 2.1 GEMV2 (Geometry-based Efficient propagation Model for V2V)

**Source**: Boban et al., IEEE T-VT 2014. [arXiv:1305.0124](https://arxiv.org/abs/1305.0124)

GEMV2 is the gold standard for V2V propagation simulation:

| Feature                 | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Link classification** | LOS, NLOSv (blocked by vehicles), NLOSb (blocked by buildings/static objects) |
| **Large-scale**         | Deterministic path loss and shadowing per link type                           |
| **Small-scale**         | Stochastic based on surrounding object density                                |
| **Validation**          | Validated against measurements in urban, suburban, highway, and open-space    |
| **Scalability**         | Runs for 10k+ vehicles in city-wide simulations                               |

**Path loss models (GEMV2):**

- LOS: Free-space + two-ray ground reflection
- NLOSv: Free-space + vehicle diffraction loss (10-20 dB additional)
- NLOSb: Log-distance with n=2.5-3.5 + building diffraction/reflection

**Pros**: Highly realistic, validated, geometry-aware.
**Cons**: Requires detailed environment geometry (buildings, foliage outlines). Heavy for real-time sim.

### 2.2 3GPP TR 37.885 (V2X Channel Model)

**Source**: 3GPP Release 15, Study on V2X evaluation methodology

Standardized channel model for C-V2X (LTE/NR sidelink):

| Scenario | Path Loss (LOS)                            | Path Loss (NLOS)                         |
| -------- | ------------------------------------------ | ---------------------------------------- |
| Urban    | PL = 38.77 + 16.7*log10(d) + 18.2*log10(f) | PL = 36.85 + 30*log10(d) + 18.9*log10(f) |
| Highway  | PL = 32.4 + 20*log10(d) + 20*log10(f)      | PL = 36.85 + 30*log10(d) + 18.9*log10(f) |

LOS probability:

- Urban: `P_LOS = min(1, 1.05 * exp(-0.0114 * d))` for d > 0
- Highway: `P_LOS = min(1, 2.1013 - 0.002 * d)` for 0 < d < 475m

**Pros**: Standardized, well-defined parameters, includes LOS probability.
**Cons**: Designed for cellular sidelink, not directly applicable to MAVLink/DSRC without adaptation.

### 2.3 Log-Distance Path Loss + Fading

The most widely used simplified model in V2V simulation:

```
PL(d) = PL(d0) + 10 * n * log10(d/d0) + X_sigma
```

Where:

- `PL(d0)` = path loss at reference distance d0 (typically 1m)
- `n` = path loss exponent (2.0 LOS, 2.5-4.0 NLOS)
- `X_sigma` = log-normal shadow fading ~ N(0, sigma^2)
- sigma = 3-4 dB (LOS), 6-8 dB (NLOS)

Small-scale fading:

- **Rician** for LOS: K-factor 3-10 dB (strong direct path)
- **Rayleigh** for NLOS: K=0 (no dominant path)

### 2.4 Two-Ray Ground Reflection

```
Pr = Pt * Gt * Gr * (ht * hr)^2 / (d^4)    (for d >> crossover distance)
```

Crossover distance: `d_c = 4 * pi * ht * hr / lambda`

Useful for highway V2V where antenna heights are similar (~1.5m).

### 2.5 Log-Ray Hybrid Model

**Source**: Arabian Journal for Science and Engineering, 2023

Combines log-distance and two-ray:

- RMSE of 2.54 dB vs 3.07 dB (log-distance) and 3.72 dB (two-ray)
- 17-32% improvement in fitting accuracy

---

## 3. Model Comparison for Our Use Case

| Criterion              | GEMV2         | 3GPP 37.885  | Log-Distance+Fading | Two-Ray      |
| ---------------------- | ------------- | ------------ | ------------------- | ------------ |
| **Realism**            | Very High     | High         | Medium-High         | Medium       |
| **Computation**        | Heavy         | Moderate     | **Light**           | Light        |
| **LOS/NLOS**           | Yes (3 types) | Yes          | **Yes**             | No           |
| **Fading**             | Stochastic    | Defined      | **Rician/Rayleigh** | No           |
| **Shadow fading**      | Deterministic | Log-normal   | **Log-normal**      | No           |
| **Environment data**   | Required      | Not required | **Not required**    | Not required |
| **Real-time capable**  | Difficult     | Yes          | **Yes**             | Yes          |
| **V2V validated**      | Yes           | Yes          | Yes                 | Partially    |
| **Integration effort** | High          | Medium       | **Low**             | Low          |

---

## 4. Recommended Model: Hybrid Log-Distance with LOS/NLOS Classification

For our multi-vehicle simulation, the optimal choice is a **hybrid model** that combines:

1. **Geometric LOS/NLOS classification** using jamming zones and obstacles as occluders
2. **Log-distance path loss** with different exponents for LOS vs NLOS
3. **Log-normal shadow fading** with correlated samples
4. **Rician/Rayleigh small-scale fading** based on link type
5. **Link quality metric** mapping received power to a 0-1 quality score

This approach provides realistic propagation without requiring detailed building geometry, and it integrates naturally with the existing jamming zone infrastructure.

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  V2VChannelModel                         │
│                                                          │
│  1. LOS/NLOS Classification                              │
│     ├── Ray-cast against obstacles (JammingZone spheres) │
│     ├── Check if any vehicle body occludes the link      │
│     └── Returns: LOS, NLOSv (vehicle), NLOSo (obstacle) │
│                                                          │
│  2. Large-Scale Path Loss                                │
│     ├── LOS:   PL = PL0 + 10*n_LOS*log10(d/d0)         │
│     ├── NLOSv: PL = PL_LOS + V_loss (vehicle loss)      │
│     └── NLOSo: PL = PL0 + 10*n_NLOS*log10(d/d0)        │
│                                                          │
│  3. Shadow Fading (log-normal, spatially correlated)     │
│     ├── LOS:   sigma = 3.0 dB                           │
│     ├── NLOSv: sigma = 5.0 dB                           │
│     └── NLOSo: sigma = 7.0 dB                           │
│                                                          │
│  4. Small-Scale Fading                                   │
│     ├── LOS:   Rician (K = 6 dB)                        │
│     └── NLOS:  Rayleigh (K = 0)                          │
│                                                          │
│  5. Link Quality Computation                             │
│     ├── P_rx = P_tx - PL + shadow + fading               │
│     ├── SNR = P_rx - noise_floor                         │
│     └── quality = sigmoid(SNR - threshold)               │
│                                                          │
│  6. Jamming Degradation (existing model preserved)       │
│     └── quality_final = quality * D_i * D_j              │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Key Parameters

| Parameter                 | LOS   | NLOSv | NLOSo | Unit |
| ------------------------- | ----- | ----- | ----- | ---- |
| Path loss exponent (n)    | 2.0   | 2.5   | 3.5   | -    |
| Reference PL at 1m (PL0)  | 47.86 | 47.86 | 47.86 | dBm  |
| Shadow fading std (sigma) | 3.0   | 5.0   | 7.0   | dB   |
| Vehicle obstruction loss  | -     | 12.0  | -     | dB   |
| Rician K-factor           | 6.0   | -inf  | -inf  | dB   |
| Tx power                  | 23.0  | 23.0  | 23.0  | dBm  |
| Noise floor               | -95.0 | -95.0 | -95.0 | dBm  |
| Carrier frequency         | 5.9   | 5.9   | 5.9   | GHz  |

### 4.3 Integration Points

The model integrates with the existing codebase at:

1. **`src/algo/utils_3d.py`** - `calculate_aij` / `calculate_gij` will be replaced by the new model for pairwise quality
2. **`src/algo/controller.py`** - Communication quality matrix computation
3. **`src/algo/base.py`** - JammingZone degradation factors are preserved and applied on top
4. **`src/algo/mavlink.py`** - Packet loss now derives from the realistic link quality
5. **`src/simulation/api.py`** - Agent-level `communication_quality` uses the new model

---

## 5. References

1. Boban, M., et al. "Geometry-Based Vehicle-to-Vehicle Channel Modeling for Large-Scale Simulation." IEEE T-VT, 2014. [arXiv:1305.0124]
2. 3GPP TR 37.885. "Study on evaluation methodology of new V2X use cases for LTE and NR." Release 15, 2019.
3. Matolak, D.W. & Sun, R. "V2V channel characteristics and models." Vehicular Technology, 2017.
4. Saifgharbii. "Analysis and Simulation of V2V Fading Channels." GitHub, 2024.
5. Al-Samman, A. et al. "A Proposed V2V Path Loss Model: Log-Ray." Arabian J. Sci. Eng., 2023.
