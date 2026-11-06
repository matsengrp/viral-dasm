"""DASM2 data loading and processing modules.

This module provides unified data representations for evolutionary modeling.

Create datasets using factory functions::

    from dasm2.data.evolutionary_dataset import (
        create_evolutionary_dataset,
        create_whichmut_dataset,
        create_dms_dataset,
    )
    bundle = create_evolutionary_dataset(df, neutral_model)

Import utilities from their specific modules::

    from dasm2.data.segment_handlers import (
        AntibodySegmentHandler,
        SegmentHandler,
    )
    from dasm2.data.registry import (
        load_dataframe_with_split_markers,
        add_train_val_markers,
    )
"""
