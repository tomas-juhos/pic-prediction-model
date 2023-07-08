from datetime import datetime


class GBMParameters:
    universe_constr: str
    testing_start: datetime
    testing_end: datetime
    model_id: int
    val_criterion: str

    max_depth: int
    num_leaves: int
    min_data_in_leaf: int
    seed: int
    #  time_steps: int
    verbose: int

    @classmethod
    def build_record(cls, key, params):
        res = cls()
        res.universe_constr = key[0]
        res.testing_start = key[1]
        res.testing_end = key[2]
        res.model_id = key[3]
        res.val_criterion = key[4]

        res.max_depth = params["max_depth"]
        res.num_leaves = params["num_leaves"]
        res.min_data_in_leaf = params["min_data_in_leaf"]
        res.seed = params["seed"]
        # res.time_steps = params['TIME_STEPS']
        res.verbose = params["verbose"]

        return res

    def as_tuple(self):
        return (
            self.universe_constr,
            self.testing_start,
            self.testing_end,
            self.model_id,
            self.val_criterion,
            self.max_depth,
            self.num_leaves,
            self.min_data_in_leaf,
            self.seed,
            #  self.time_steps,
            #  self.verbose,
        )
