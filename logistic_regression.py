import numpy
import warnings

class LogisticRegression:
    warnings.filterwarnings('ignore')
    
    def __init__(self, alpha=0.001, iteration=20000, split_ratio=0.75):
        self.alpha = alpha
        self.iteration = iteration
        self.ratio = split_ratio
    
    
    def train_test_split(self, X, y):
        mask = np.random.rand(len(self.X)) < self.ratio
        
        return self.X[mask], self.y[mask], self.X[~mask], self.y[~mask]
        
        
    def fit(self, X, y):
        self.X = np.insert(np.array(X), 0, 1, axis=1)
        self.y = np.array(y)
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split(self.X, self.y)
        
        row, column = self.X_train.shape
        self.theta = np.zeros(column)
        
        
        for _ in range(self.iteration):
            self.training_model(row)
            
    
    def training_model(self, row):
        y_hat = 1 / (1 + np.exp( - (self.X_train @ self.theta) ))
        self.theta = self.theta - self.alpha * ( ( (y_hat - self.y_train) @ self.X_train ) / row)
            
            
    def predict(self):
        self.y_pred = 1 / (1 + np.exp( - (self.X_test @ self.theta) ))
        self.y_pred = np.where(self.y_pred > 0.5, 1, 0)
        
        return self.y_pred
        
        
    def accuracy(self):
        compare = self.y_pred == self.y_test
        
        return len([i for i in compare if i == True]) / len(self.X_test) * 100
    
    
