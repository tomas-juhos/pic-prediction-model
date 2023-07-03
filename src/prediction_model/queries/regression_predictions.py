"""Regression predictions queries."""


class Queries:
    """Regression Predictions queries class."""

    UPSERT = (
        "INSERT INTO regression_predictions ("
        "       model, "
        "       train_criterion, "
        "       val_criterion, "
        "       datadate, "
        "       gvkey, "
        "       predicted_rtn, "
        "       real_rtn, "
        "       dir_acc, "
        "       chosen_bottom, "
        "       chosen_weighted "
        ") VALUES %s "
        "ON CONFLICT (model, train_criterion, val_criterion, datadate, gvkey) DO "
        "UPDATE SET "
        "       model=EXCLUDED.model, "
        "       train_criterion=EXCLUDED.train_criterion, "
        "       val_criterion=EXCLUDED.val_criterion, "
        "       datadate=EXCLUDED.datadate, "
        "       gvkey=EXCLUDED.gvkey, "
        "       predicted_rtn=EXCLUDED.predicted_rtn, "
        "       real_rtn=EXCLUDED.real_rtn, "
        "       dir_acc=EXCLUDED.dir_acc, "
        "       chosen_bottom=EXCLUDED.chosen_bottom, "
        "       chosen_weighted=EXCLUDED.chosen_weighted; "
    )
