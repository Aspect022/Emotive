# EMOTIV Device Characterization
Generated: 2026-08-14T15:59:30.123914+00:00

## Device Identification

| Field | Value | Evidence |
|---|---|---|
| **Headset Model** | EPOC X | Filename suffix `_EPOCX_`, JSON `headsetType: "EPOCX"`, 14-electrode schema |
| **Confidence** | HIGH | Consistent across all files |
| **Electrode Count** | 14 | Confirmed from CSV column headers |
| **Electrode Names** | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 | CSV headers |
| **Montage** | UNKNOWN (reported in JSON) — Standard EPOC X positions | Confirmed by electrode names matching 10-20 system |
| **EEG Sampling Rate** | 128 Hz (md.bp.csv) or higher (md.csv) | CSV metadata line |
| **Band Power Rate** | 8 Hz (POW.* columns) | EMOTIV documentation |
| **Motion Rate** | 32 Hz | MOT.* columns + metadata `mot_32` |
| **Connection** | Bluetooth | JSON `connectionType: "bluetooth"` |
| **Export Software** | EmotivPRO (`com.emotiv.emotivpro`) | JSON `exportApp` |
| **Reference** | CMS/DRL (P3/P4 positions) | Standard EPOC X hardware design |

## EPOC X Technical Specifications (from EMOTIV documentation)

| Specification | Value |
|---|---|
| Electrode type | Saline-based felt pads (wet) |
| Channels | 14 EEG + 2 reference (CMS/DRL) |
| ADC resolution | 16-bit |
| EEG bandwidth | 0.16–43 Hz (hardware filtered) |
| Notch filter | 50 Hz and 60 Hz |
| Dynamic range | ±420 µV |
| Noise floor | <1 µV RMS |
| Output units | µV |
| Sample rate options | 128 Hz / 256 Hz |
| Battery | 12–14 hours Li-poly |
| Connectivity | Bluetooth 4.2 |

## Important Constraints for Analysis

1. **Low spatial resolution**: 14 channels cannot support high-density source localization.
2. **Hardware filtering already applied**: The 0.16 Hz high-pass and 43 Hz low-pass are fixed.
   Additional software filtering should respect these limits.
3. **Saline electrodes**: Impedance degrades over time without re-moistening.
   Contact quality metrics (CQ.*) must be monitored.
4. **Bluetooth latency**: Possible sample-level jitter in wireless transmission.
   Timestamps in software may not reflect exact hardware sample time.
5. **CMS/DRL reference**: Re-referencing to average is standard practice for EPOC X.
6. **Interpolated samples**: EMOTIV flags interpolated samples (`EEG.Interpolated = 1`).
   These must be tracked and excluded from artifact-sensitive analyses.

## Streams Available per Export File

| Stream | Columns | Rate | Notes |
|---|---|---|---|
| EEG (raw) | `EEG.AF3` ... `EEG.AF4` | 128/256 Hz | Primary signal |
| Contact Quality | `CQ.AF3` ... `CQ.Overall` | 128/256 Hz | 0=no contact; 4=excellent |
| EEG Quality | `EQ.*` | 128/256 Hz | Derived quality metric |
| Motion/IMU | `MOT.Q0`..`MOT.MagZ` | 32 Hz | Head movement detection |
| Band Power | `POW.*.<Theta/Alpha/BetaL/BetaH/Gamma>` | 8 Hz | PROPRIETARY derived metric |
| Markers | `MarkerIndex`, `MarkerType`, `MarkerValueInt` | Per event | Experimenter events |
| Counter | `EEG.Counter` | 128/256 Hz | Rollover counter (0–255) |

## Scientific Caveat on Proprietary Metrics

The `POW.*` band-power columns are **EMOTIV proprietary**, computed with undisclosed algorithms.
They must NOT be treated as validated psychological constructs.
**Analysis should primarily use the raw EEG channels** (EEG.AF3 through EEG.AF4).
