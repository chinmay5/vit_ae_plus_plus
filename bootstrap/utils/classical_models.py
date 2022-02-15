
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class MLModel:
    def __init__(self):
        pass

    def module_name(self):
        raise NotImplementedError

    def execute_method(self, train_features, train_label, test_features):
        raise NotImplementedError


class SVMModel(MLModel):
    def __init__(self):
        super(SVMModel, self).__init__()

    def module_name(self):
        return "svm"

    def execute_method(self, train_features, train_label, test_features):
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear']}

        grid = RandomizedSearchCV(svm.SVC(probability=True, random_state=42, class_weight='balanced'), param_grid, refit=True, verbose=0,
                                  random_state=42)
        # fitting the model for grid search
        grid.fit(train_features, train_label)
        # print best parameter after tuning
        print(grid.best_params_)

        # print how our model looks after hyper-parameter tuning
        print(grid.best_estimator_)
        pred = grid.predict_proba(test_features)
        return pred


class RFModel(MLModel):
    def execute_method(self, train_features, train_label, test_features):
        param_grid = {'n_estimators': [50, 100, 150, 200],
                      'criterion': ["gini", "entropy"],
                      'class_weight': ["balanced", "balanced_subsample"]}

        grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=0,
                                  random_state=42)
        grid.fit(train_features, train_label)
        print(grid.best_params_)
        print(grid.best_estimator_)
        pred = grid.predict_proba(test_features)
        return pred

    def __init__(self):
        super(RFModel, self).__init__()

    def module_name(self):
        return "rf"


class LinearModel(MLModel):
    def __init__(self):
        super(LinearModel, self).__init__()

    def execute_method(self, train_features, train_label, test_features):
        param_grid = {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                      'penalty': ["l2", "none"],
                      'max_iter': [200, 500, 1000]
                      }
        grid = RandomizedSearchCV(LogisticRegression(random_state=42, class_weight='balanced'), param_grid, refit=True,
                                  verbose=0, random_state=42)
        # fitting the model for grid search
        grid.fit(train_features, train_label)
        print(grid.best_params_)
        print(grid.best_estimator_)
        pred = grid.predict_proba(test_features)
        return pred

    def module_name(self):
        return "linear"


def execute_models(train_features, train_label, test_features, *methods):
    model_map = {x().module_name(): x for x in MLModel.__subclasses__()}
    results = {}
    for method in methods:
        assert method in model_map.keys(), "Invalid choice of execution method"
        # MyClass = getattr(importlib.import_module(os.path.join(PROJECT_ROOT_DIR, "bootstrap", "util")), "MyClass")
        results[method] = model_map[method]().execute_method(train_features=train_features, train_label=train_label, test_features=test_features)
    return results
