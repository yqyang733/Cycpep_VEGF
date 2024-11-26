class Config():
    def __init__(self):
        
        # global settings
        self.trajpath = "../all_traj/"
        self.refstructure = "complex.psf"
        self.traj = "com-prodstep.dcd"
        self.partA = ["M",]     # chain ID
        self.partB = ["V",]     # chain ID
        self.startframe = 1
        self.endframe = 3
        self.step = 1
        self.get_des = "all"   # all or rank
        self.rank = 10
        self.channels = 1

        # data path
        self.trainlist = "trainlst.dat"
        self.predictlist = "predictlst.dat"

        # select descriptors
        self.pickdescriptorsways = "frequency"  # frequency or std or forward
        self.descriptornums = -1

        # Model selection
        self.model = "GradientBoostingRegressor"   # GradientBoostingRegressor or 

        # GradientBoosting
        self.n_estimators = 100
        self.max_depth = 3
        self.learning_rate = 0.05

        self.epoch = 50

        # MLP
        self.n_hidden = 8





