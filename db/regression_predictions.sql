CREATE TABLE regression_predictions (
    model                   VARCHAR(20),
    train_criterion         VARCHAR(20),
    val_criterion           VARCHAR(20),

    datadate                TIMESTAMP,
    gvkey                   INTEGER,

    predicted_rtn           DECIMAL(18,4),
    real_rtn                DECIMAL(18,4),

    chosen_bottom           BOOLEAN,
    chosen_weighted         BOOLEAN,

    PRIMARY KEY (model, train_criterion, val_criterion, datadate, gvkey)
);