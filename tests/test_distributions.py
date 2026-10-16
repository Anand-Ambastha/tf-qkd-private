# -*- coding: utf-8 -*-
"""PDF normalization and CDF monotonicity/endpoint checks for every model."""
import numpy as np
import pytest
from src.validation import check_pdf_normalization, check_cdf_monotonic


def test_pdf_integrates_to_one(all_stochastic_channels):
    for name, ch in all_stochastic_channels:
        result = check_pdf_normalization(ch, atol=5e-3)
        assert result['ok'], f"{name}: PDF integral = {result['integral']}, expected ~1"


def test_cdf_monotonic_and_approaches_one(all_stochastic_channels):
    for name, ch in all_stochastic_channels:
        result = check_cdf_monotonic(ch)
        assert result['monotonic'], f"{name}: CDF is not monotonic"
        assert result['ok_endpoint'], f"{name}: CDF endpoint = {result['endpoint']}, expected ~1"


def test_cdf_zero_at_origin(all_stochastic_channels):
    for name, ch in all_stochastic_channels:
        val = float(np.atleast_1d(ch.cdf(np.array([1e-12])))[0])
        assert val < 1e-2, f"{name}: CDF near 0 should be ~0, got {val}"


def test_deterministic_channel_point_mass(deterministic_channel):
    ch = deterministic_channel
    assert 0.0 < ch.eta_nominal <= 1.0
    assert float(ch.cdf(np.array([ch.eta_nominal - 1e-9]))[0]) == 0.0
    assert float(ch.cdf(np.array([ch.eta_nominal + 1e-9]))[0]) == 1.0
    samples = ch.sample(100)
    assert np.allclose(samples, ch.eta_nominal)
