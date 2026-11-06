"""Neutral evolutionary models for DASM2.

This module provides implementations of neutral mutation models used to compute
expected mutation probabilities under neutral evolution (no selection).

Available Models:
    - ThriftySHMModel: Native DASM2 implementation of Thrifty SHM rate prediction
    - HitClassModel: Model for applying hit class corrections
    - MultihitWrapper: Adds multihit correction to rates/CSPs-based neutral models
    - TabularRatesNeutralModel: Context-dependent rates from tabular data
    - S5FNeutralModel: Five-mer SHM rates/CSPs via a pure lookup table
    - GTRCodonNeutralModel: Context-free rates via GTR + matrix exponentiation
    - ShmCodonNeutralModel: Context-free, non-reversible codon CTMC (SHM-1)
    - PrecomputedRatesNeutralModel: Rate-free carrier that makes a multihit
      correction configurable on corpora whose rates are already in the data

Model Hierarchy:
    - AbstractNeutralModel: Base class for all neutral models
    - RatesCSPNeutralModel: For models using rates/CSPs decomposition
      (ThriftySHM, TabularRates, S5F, MultihitWrapper, PrecomputedRates)
    - GTRCodonNeutralModel inherits directly from AbstractNeutralModel
      (uses matrix exponentiation instead of rates/CSPs)

Import neutral models directly from specific modules:
    - from dasm2.neutral.base import AbstractNeutralModel, RatesCSPNeutralModel
    - from dasm2.neutral.thrifty import ThriftySHMModel
    - from dasm2.neutral.hit_class import HitClassModel
    - from dasm2.neutral.multihit import MultihitWrapper
    - from dasm2.neutral.tabular_rates import TabularRatesNeutralModel
    - from dasm2.neutral.s5f import S5FNeutralModel
    - from dasm2.neutral.gtr_codon import GTRCodonNeutralModel
    - from dasm2.neutral.shm_codon import ShmCodonNeutralModel
    - from dasm2.neutral.precomputed_rates import PrecomputedRatesNeutralModel
"""
