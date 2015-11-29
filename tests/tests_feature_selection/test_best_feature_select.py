import unittest
from mlxtend.feature_selection.best_combination_feature_selection import BestFeature

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testBest(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        irsi = load_iris()
        X, y = irsi.data, irsi.target
        clf = RandomForestClassifier()
        bestFeature = BestFeature(clf)
        bestFeature.fit(X, y)
        

if __name__ == "__main__":
    unittest.main()