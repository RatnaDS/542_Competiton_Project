from ml_utils import Model
from sklearn.svm import SVC


class SKLearnSVMModel(Model):

    def __init__(self):
        
        super(SKLearnSVMModel, self).__init__()
        self.model = SVC()
        
    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, y):
        return self.model.fit(x, y)
