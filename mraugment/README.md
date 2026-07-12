# MRAugment

Self-contained integration of the physics-aware augmentation pipeline from
Fabian, Heckel, and Soltanolkotabi, *Data augmentation for deep learning based
accelerated MRI reconstruction with limited data* (ICML 2021).

The full complex multi-coil image is augmented before undersampling. Full
k-space and the RSS target are then regenerated from the same image, ensuring
that the model input and label remain physically consistent. Validation and
leaderboard data are never augmented.

Enable the recommended low-data configuration with:

```bash
python train.py --mraugment --aug-strength 0.55
```

The default exponential schedule starts near zero and reaches the requested
strength at the final epoch. Use `--aug-schedule constant` to apply the full
strength immediately. Run `python train.py --help` for all transform limits
and probability weights.

This implementation is adapted from the authors' MIT-licensed reference:
<https://github.com/AIF4S/MRAugment>.
