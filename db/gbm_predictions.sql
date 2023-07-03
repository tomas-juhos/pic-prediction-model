CREATE TABLE gbm_predictions (
    model_id                BIGINT,
    val_criterion           VARCHAR(20),

    datadate                TIMESTAMP,
    gvkey                   INTEGER,

    predicted_rtn           DECIMAL(18,4),
    real_rtn                DECIMAL(18,4),

    dir_acc                 INTEGER,

    chosen_bottom           BOOLEAN,
    chosen_weighted         BOOLEAN,

    PRIMARY KEY (model_id, val_criterion, datadate, gvkey)
);