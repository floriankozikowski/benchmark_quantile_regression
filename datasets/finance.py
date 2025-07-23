from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "finance"
    install_cmd = "pip"
    requirements = ["libsvmdata"]

    @staticmethod
    def _load_finance_data():
        # Load the finance dataset using libsvmdata
        # == E2006-log1p
        X, y = fetch_libsvm("finance")
        return X, y

    def get_data(self):
        try:
            X, y = self.X, self.y
        except AttributeError:
            X, y = self._load_finance_data()
            self.X, self.y = X, y
        return dict(X=X, y=y)
