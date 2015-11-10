'''
Created on Nov 10, 2558 BE

@author: ECP
original : http://sebastianraschka.com/Articles/2014_sequential_sel_algos.html
git : https://github.com/rasbt/mlxtend#sequential-backward-selection

'''
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SBS
from sklearn.datasets import load_iris

def test_hello():
    print 'test'
    iris = load_iris()
    x, y = iris.data, iris.target
    print x[0]
    clf = DecisionTreeClassifier()
    sbs = SBS(clf, k_features=2)
    sbs = sbs.fit(x, y)
    print sbs.transform(x)
            
if __name__ == '__main__':
    test_hello()