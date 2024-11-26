from sklearn.ensemble import GradientBoostingRegressor

class GradientBoosting():

    def __init__(self, n_estimators, max_depth, learning_rate):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def build(self):
        
        return GradientBoostingRegressor(n_estimators = self.n_estimators, max_depth = self.max_depth, learning_rate = self.learning_rate)