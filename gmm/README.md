# gmm/

Single-node detection primitives shared by the Python simulator and the C++
deployment. Imported by `simulation/`; not a standalone entry point.

```
config.py     shared constants (mirrored by deployment/config.h)
features.py   load_log_mel, gwrp_weights, extract_feature_r
gmm.py        DiagGMM (numpy, mirrors deployment/gmm.h)
detector.py   GMMDetector: fit + calibrate + score + save/load
```
