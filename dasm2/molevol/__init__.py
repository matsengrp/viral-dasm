"""Molecular evolution functions organized by purpose.

Molecular evolution functions are organized into focused modules:
- tensor_ops: Pure tensor manipulation utilities for evolution calculations
- mutation_matrices: Core mutation matrix construction and codon probability calculation
- codon_conversion: Converting between codon and amino acid representations
- neutral_evolution: Neutral evolution probability calculations
- mutsel: Mutation-selection evolution modeling
- branch_optimization: Branch length optimization utilities

Import functions directly from specific modules:
- from dasm2.molevol.tensor_ops import reshape_for_codons, flatten_codons
- from dasm2.molevol.neutral_evolution import neutral_codon_probs
- from dasm2.molevol.mutsel import compute_log_adjusted_codon_probs
"""

# No imports - use explicit imports from specific modules

__doc__ = """Free functions for molecular evolution computation.

CSP means conditional substitution probability. CSPs are the probabilities of alternate
states conditioned on there being a substitution.

Functions are organized into focused modules - import from specific modules:
- tensor_ops: Pure tensor utilities (reshape_for_codons, flatten_codons, unflatten_codons, set_parent_codon_prob)
- mutation_matrices: Core mechanics (build_mutation_matrices, codon_probs_of_mutation_matrices)
- codon_conversion: Conversion functions (aaprobs_of_codon_probs, codon_probs_of_parent_scaled_nt_rates_and_csps)
- neutral_evolution: Neutral probabilities (neutral_codon_probs, neutral_aa_probs, neutral_aa_mut_probs, etc.)
- mutsel: Mutation-selection models (compute_log_adjusted_codon_probs, compute_dasm_log_pcp_probability, mutsel_log_pcp_probability_of, etc.)
- branch_optimization: Branch length optimization with warm-started bounded scipy
"""
