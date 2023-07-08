"""GBM Parameters queries."""


class Queries:
    """GBM Parameters queries class."""

    UPSERT = (
        "INSERT INTO gbm_parameters ("
        "       universe_constr, "
        "       testing_start, "
        "       testing_end, "
        "       model_id, "
        "       val_criterion, "
        "       max_depth, "
        "       num_leaves, "
        "       min_data_in_leaf, "
        "       seed "
        ") VALUES %s "
        "ON CONFLICT (universe_constr, testing_start, testing_end, model_id, val_criterion) DO "
        "UPDATE SET "
        "       universe_constr=EXCLUDED.universe_constr, "
        "       testing_start=EXCLUDED.testing_start, "
        "       testing_end=EXCLUDED.testing_end, "
        "       model_id=EXCLUDED.model_id, "
        "       val_criterion=EXCLUDED.val_criterion, "
        "       max_depth=EXCLUDED.max_depth, "
        "       num_leaves=EXCLUDED.num_leaves, "
        "       min_data_in_leaf=EXCLUDED.min_data_in_leaf, "
        "       seed=EXCLUDED.seed; "
    )
