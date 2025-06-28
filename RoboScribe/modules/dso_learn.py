from copy import deepcopy

from dso import DeepSymbolicOptimizer

class DSOLearner(DeepSymbolicOptimizer):
    def __init__(self, config=None):
        if config is None:
            config = {
                "task": {"task_type": "regression"}
            }
        DeepSymbolicOptimizer.__init__(self, config)

    def fit(self, X, y):
        # Update the Task
        config = deepcopy(self.config)
        config["task"]["dataset"] = (X, y)

        # Turn off file saving
        config["experiment"]["logdir"] = None
        config["gp_meld"]["run_gp_meld"] = False

        self.set_config(config)

        train_result = self.train()
        self.program_ = train_result["program"]

        return self.program_

    def predict(self, X):

        return self.program_.execute(X)
    
    def test_setup(self, X, y):
        # Update the Task
        config = deepcopy(self.config)
        config["task"]["dataset"] = (X, y)

        # Turn off file saving
        config["experiment"]["logdir"] = None
        config["gp_meld"]["run_gp_meld"] = False

        self.set_config(config)
        self.setup()


    def save_model(self, save_path):
        self.save(save_path)

    def load_model(self, load_path):
        self.load(load_path)