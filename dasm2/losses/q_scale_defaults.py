"""Canonical QDASM/Hybrid Q-scale defaults.

These scalars parameterize the Q-family losses. The default triple was
calibrated against Thrifty; fitted triples are properties of a (neutral model,
corpus) pair and therefore are named separately rather than made variant-based
defaults.

Calibrated by held-out LL maximisation under fixed pre-trained Thrifty rates
(issue #589). ``dasm2/configs/qdasm_v4_4m.yaml`` and ``hybrid_4m.yaml`` state
the same literal values so serialized configs remain self-contained.
"""

DEFAULT_SINGLE_HIT_SCALE = 0.566
DEFAULT_DOUBLE_HIT_SCALE = 0.02457
DEFAULT_TRIPLE_HIT_SCALE = 0.004002

# S5F-HH fit on the pooled 30k Tang+SHMOOF out-of-frame subset (#959 work item
# 3). Shipping config: dasm2/configs/qdasm_s5f_4m.yaml.
S5F_HH_POOLED_OOF_SINGLE_HIT_SCALE = 0.6546
S5F_HH_POOLED_OOF_DOUBLE_HIT_SCALE = 0.038666
S5F_HH_POOLED_OOF_TRIPLE_HIT_SCALE = 0.010061
