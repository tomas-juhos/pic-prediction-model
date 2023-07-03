CREATE TABLE regression_metrics (
    testing_start       DATE,
    testing_end         DATE,

    model               VARCHAR(20),
    train_criterion     VARCHAR(20),
    val_criterion       VARCHAR(20),

    rtn_bottom          DECIMAL(18,4),
    rtn_weighted        DECIMAL(18,4),
    rtn_random          DECIMAL(18,4),
    rtn_benchmark       DECIMAL(18,4),

    mse                 DECIMAL(18,4),
    rmse                DECIMAL(18,4),
    mae                 DECIMAL(18,4),
    mape                DECIMAL(18,4),
    dir_acc             DECIMAL(5,2),
    f_pvalue            DECIMAL(5,2),
    r_sqr               DECIMAL(5,2),

    training_start      DATE,
    training_end        DATE,
    validation_start    DATE,
    validation_end      DATE,

    PRIMARY KEY (testing_start, testing_end, model, train_criterion, val_criterion)
);