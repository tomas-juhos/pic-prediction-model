CREATE TABLE regression_predictions (
    universe_constr         VARCHAR(50),

    model                   VARCHAR(20),
    train_criterion         VARCHAR(20),
    val_criterion           VARCHAR(20),

    datadate                TIMESTAMP,
    gvkey                   INTEGER,

    predicted_rtn           DECIMAL(18,4),
    real_rtn                DECIMAL(18,4),

    dir_acc                 INTEGER,

    chosen_bottom           BOOLEAN,
    chosen_weighted         BOOLEAN,
    chosen_random           BOOLEAN,

    PRIMARY KEY (universe_constr, model, train_criterion, val_criterion, datadate, gvkey)
);