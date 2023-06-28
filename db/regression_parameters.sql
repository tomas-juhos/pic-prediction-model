CREATE TABLE regression_parameters (
    testing_start               DATE,
    testing_end                 DATE,

    model                       VARCHAR(20),
    train_criterion             VARCHAR(20),
    val_criterion               VARCHAR(20),

    alpha                       DECIMAL(20,6),

    utilization_pct             DECIMAL(20,6),
    bar                         DECIMAL(20,6),
    age                         DECIMAL(20,6),
    tickets                     DECIMAL(20,6),
    units                       DECIMAL(20,6),
    market_value_usd            DECIMAL(20,6),
    loan_rate_avg               DECIMAL(20,6),
    loan_rate_max               DECIMAL(20,6),
    loan_rate_min               DECIMAL(20,6),
    loan_rate_range             DECIMAL(20,6),
    utilization_pct_delta       DECIMAL(20,6),
    bar_delta                   DECIMAL(20,6),
    age_delta                   DECIMAL(20,6),
    tickets_delta               DECIMAL(20,6),
    units_delta                 DECIMAL(20,6),
    market_value_usd_delta      DECIMAL(20,6),
    loan_rate_avg_delta         DECIMAL(20,6),
    loan_rate_max_delta         DECIMAL(20,6),
    loan_rate_min_delta         DECIMAL(20,6),
    loan_rate_range_delta       DECIMAL(20,6),
    short_interest              DECIMAL(20,6),
    short_ratio                 DECIMAL(20,6),
    market_cap                  DECIMAL(20,6),
    shares_out                  DECIMAL(20,6),

    PRIMARY KEY (testing_start, testing_end, model, train_criterion, val_criterion)
);