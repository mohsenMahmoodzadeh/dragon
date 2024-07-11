from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb


def create_cls_model_spec(task, data_spec):
    cls_model_spec = {
        'knn': {
            'cls': make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=len(data_spec[task]['label_names']))
            )
        },
        'svc': {
            'cls': make_pipeline(StandardScaler(), SVC(gamma='auto'))
        },

        'xgboost': {
            'cls': xgb.XGBClassifier()
        }
    }

    return cls_model_spec
