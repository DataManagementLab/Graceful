import copy

import numpy as np
from sklearn.metrics import mean_squared_error


class Metric:
    """
    Abstract class defining a metric used to evaluate the zero-shot cost model performance (e.g., Q-error)
    """

    def __init__(self, metric_prefix='val_', metric_name='metric', maximize=True, early_stopping_metric=False):
        self.maximize = maximize
        self.default_value = -np.inf
        if not self.maximize:
            self.default_value = np.inf
        self.best_seen_value = self.default_value
        self.last_seen_value = self.default_value
        self.metric_prefix = metric_prefix
        self.metric_name = metric_prefix + metric_name
        self.best_model = None
        self.early_stopping_metric = early_stopping_metric

    def evaluate(self, model=None, metrics_dict=None, update_best_seen: bool = True, **kwargs):
        metric = self.default_value
        try:
            metric = self.evaluate_metric(**kwargs)
        except ValueError as e:
            print(f"Observed ValueError in metrics calculation {e}")

        metrics_dict[self.metric_name] = metric

        if update_best_seen:
            self.last_seen_value = metric

            print(f"{self.metric_name}: {metric:.4f} [best: {self.best_seen_value:.4f}]")

            best_seen = False
            if (self.maximize and metric > self.best_seen_value) or (
                    not self.maximize and metric < self.best_seen_value):
                self.best_seen_value = metric
                best_seen = True
                if model is not None:
                    self.best_model = copy.deepcopy(model.state_dict())
            return best_seen
        else:
            print(f"{self.metric_name}: {metric:.4f}")
            return False


class MAPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mape', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mape = np.mean(np.abs((labels - preds) / labels))
        return mape

    def evaluate_metric(self, labels=None, preds=None):
        raise NotImplementedError


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mse', maximize=False, **kwargs)
        self.name = 'mse'

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        val_mse = np.sqrt(mean_squared_error(labels, preds))
        return val_mse


class MAPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mape', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mape = np.mean(np.abs((labels - preds) / labels))
        return mape


class QError(Metric):
    def __init__(self, percentile=50, min_val=0.01, verbose: bool = True, avg: bool = False, **kwargs):
        super().__init__(metric_name=f'median_q_error_{percentile}', maximize=False, **kwargs)
        self.percentile = percentile
        self.min_val = min_val
        self.verbose = verbose
        self.avg = avg
        if avg:
            self.name = 'QError(avg)'
        elif percentile == 50:
            self.name = 'QError(median)'
        elif percentile == 100:
            self.name = 'QError(max)'
        else:
            self.name = f'QError({percentile}th percentile)'

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        if not np.all(labels >= self.min_val):
            print(f"WARNING: some labels are smaller than min_val ({self.min_val}): {labels[labels < self.min_val]}",
                  flush=True)
            raise Exception(f"Labels are smaller than min_val {self.min_val}")

        if not np.all(preds >= self.min_val):
            if self.verbose:
                print(
                    f"WARNING: some preds are smaller than min_val ({self.min_val}): {preds[preds < self.min_val]} (labels: {labels[preds < self.min_val]})")
        # preds = np.abs(preds)
        preds = np.clip(preds, self.min_val, np.inf)

        q_errors = np.maximum(labels / preds, preds / labels)
        q_errors = np.nan_to_num(q_errors, nan=np.inf)
        if self.avg:
            median_q = np.mean(q_errors)
        elif self.percentile == 100:
            median_q = np.max(q_errors)
        else:
            median_q = np.percentile(q_errors, self.percentile)
        return median_q


class ProcentualError(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='procentual_error', maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        return np.mean(np.abs(preds - labels) / labels)
