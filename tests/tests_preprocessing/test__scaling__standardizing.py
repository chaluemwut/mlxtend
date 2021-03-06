from mlxtend.preprocessing import standardizing
import pandas as pd
import numpy as np


def test_standardizing_columnerror():
    try:
        ary = np.array([[1, 2], [3, 4]])
        out = standardizing(ary, [1, 's2'])
    except AttributeError:
        pass
    else:
        raise AssertionError


def test_standardizing_arrayerror():
    try:
        ary = [[1, 2], [3, 4]]
        out = standardizing(ary, [1, 's2'])
    except AttributeError:
        pass
    else:
        raise AssertionError


def test_pandas_standardizing():
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
    s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
    df = pd.DataFrame(s1, columns=['s1'])
    df['s2'] = s2

    df_out1 = standardizing(df, ['s1', 's2'])
    ary_out1 = np.array([[-1.46385, 1.46385],
                         [-0.87831, 0.87831],
                         [-0.29277, 0.29277],
                         [0.29277, -0.29277],
                         [0.87831, -0.87831],
                         [1.46385, -1.46385]])
    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)


def test_numpy_standardizing():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    ary_actu = standardizing(ary, columns=[0, 1])
    ary_expc = np.array([[-1.46385, 1.46385],
                         [-0.87831, 0.87831],
                         [-0.29277, 0.29277],
                         [0.29277, -0.29277],
                         [0.87831, -0.87831],
                         [1.46385, -1.46385]])

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_numpy_single_feat():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    ary_actu = standardizing(ary, [1])
    ary_expc = np.array([[1.46385],
                         [0.87831],
                         [0.29277],
                         [-0.29277],
                         [-0.87831],
                         [-1.46385]])

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)


def test_numpy_inplace():
    ary = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])

    standardizing(ary, [1])

    ary = ary_expc = np.array([[1, 1.46385],
                               [2, 0.87831],
                               [3, 0.29277],
                               [4, -0.29277],
                               [5, -0.87831],
                               [6, -1.46385]])

    np.testing.assert_allclose(ary, ary_expc, rtol=1e-03)


def test_numpy_single_dim():
    ary = np.array([1, 2, 3, 4, 5, 6])

    ary_actu = standardizing(ary, [0])
    ary_expc = np.array([[-1.46385],
                         [-0.87831],
                         [-0.29277],
                         [0.29277],
                         [0.87831],
                         [1.46385]])

    np.testing.assert_allclose(ary_actu, ary_expc, rtol=1e-03)
