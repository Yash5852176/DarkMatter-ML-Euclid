"""
Synthetic Euclid-like convergence map generator.

Generates mock weak-lensing shear fields using log-normal random fields that
approximate the non-Gaussian convergence κ maps expected from the ESA Euclid
wide survey.  The approach follows Hilbert et al. (2011) and Xavier et al.
(2016): we draw a Gaussian random field whose power spectrum matches a
theoretical convergence P(ℓ), then exponentiate to obtain the log-normal
realisation.

Usage:
    python -m data.mock_data_generator --n-maps 1000 --size 128 --out data/maps
"""
