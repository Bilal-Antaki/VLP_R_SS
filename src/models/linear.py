from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_linear_model(**kwargs):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression(**kwargs))
    ])