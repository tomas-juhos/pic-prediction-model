"""GBM predictions queries."""


class Queries:
    """GBM Predictions queries class."""

    UPSERT = (
        "INSERT INTO gbm_predictions ("
        "       model_id, "
        "       val_criterion, "
        "       datadate, "
        "       gvkey, "
        "       predicted_rtn, "
        "       real_rtn, "
        "       dir_acc, "
        "       chosen_bottom, "
        "       chosen_weighted, "
        "       chosen_random "
        ") VALUES %s "
        "ON CONFLICT (model_id, val_criterion, datadate, gvkey) DO "
        "UPDATE SET "
        "       model_id=EXCLUDED.model_id, "
        "       val_criterion=EXCLUDED.val_criterion, "
        "       datadate=EXCLUDED.datadate, "
        "       gvkey=EXCLUDED.gvkey, "
        "       predicted_rtn=EXCLUDED.predicted_rtn, "
        "       real_rtn=EXCLUDED.real_rtn, "
        "       dir_acc=EXCLUDED.dir_acc, "
        "       chosen_bottom=EXCLUDED.chosen_bottom, "
        "       chosen_weighted=EXCLUDED.chosen_weighted, "
        "       chosen_random=EXCLUDED.chosen_random; "
    )
