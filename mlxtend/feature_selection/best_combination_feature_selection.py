from mlxtend.feature_selection.sequential_forward_select import SFS
import numpy as np

class BestFeature(object):

    def __init__(self, estimator):
        self.estimator = estimator            
        
    
    def fit(self, X, y):
        feature_number = len(X[0])
        self.best_indices_ = []
        self.best_k_score_ = []
        for i in range(1, feature_number + 1):
            print 'feature number ', i
            sfs = SFS(estimator=self.estimator, scoring='f1_weighted', k_features=i)
            sfs = sfs.fit(X, y)
            self.best_indices_.append(sfs.indices_)
            self.best_k_score_.append(sfs.k_score_)            
            print sfs.k_score_
            print sfs.indices_
            
        best = np.argmax(self.best_k_score_)
        print 'best ', best, ' score ',self.best_k_score_[best]
        print 'index feature : ',self.best_indices_[best]