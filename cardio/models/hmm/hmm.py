""" HMModel """

import numpy as np
import dill
import warnings

from ...dataset.dataset.models.base import BaseModel

def prepare_hmm_input(batch, model, features, channel_ix):
    """Concatenate selected channel ``channel_ix`` of the batch attribute
    ``features``.

    Parameters
    ----------
    batch : EcgBatch
        Batch to concatenate.
    model : BaseModel
        A model to get the resulting arguments.
    features : str
        Specifies batch attribute that contains features for ``HMModel``.
    channel_ix : int
        Index of channel, which data should be used in training and
        predicting.

    Returns
    -------
    kwargs : dict
        Named argments for model's train or predict method. Has the
        following structure:
        "X" : 2-D ndarray
            Features concatenated along -1 axis and transposed.
        "lengths" : list
            List of lengths of individual feature arrays along -1 axis.
    """
    _ = model
    hmm_features = batch.get_variable("hmm_features")
    x = np.concatenate([features[channel_ix].T for features in hmm_features])
    lengths = [hmm_features.shape[2] for feature in hmm_features]
    return {"X": x.reshape(-1, 1), "lengths": lengths}


class HMModel(BaseModel):
    """
    Hidden Markov Model.

    This implementation is based on ``hmmlearn`` API. It is supposed
    that estimators of ``HMModel`` are model classes of ``hmmlearn``.
    """

    def __init__(self, *args, **kwargs):
        self.estimator = None
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """
        Set up estimator as an attribute and make initial settings.

        Uses estimator from model config variable as estimator.
        If config contains key ``init_param``, sets up initial
        values ``means_``, ``covars_``, ``transmat_`` and ``startprob_``
        of the estimator as defined in ``init_params``.
        """
        _ = args, kwargs
        self.estimator = self.config.get('estimator')
        init_params = self.config.get('init_params')
        if init_params is not None:
            if "m" not in self.estimator.init_params:
                self.estimator.means_ = init_params["means_"]
            if "c" not in self.estimator.init_params:
                self.estimator.covars_ = init_params["covars_"]
            if "t" not in self.estimator.init_params:
                self.estimator.transmat_ = init_params["transmat_"]
            if "s" not in self.estimator.init_params:
                self.estimator.startprob_ = init_params["startprob_"]

    def save(self, path, *args, **kwargs):  # pylint: disable=arguments-differ
        """Save ``HMModel`` with ``dill``.

        Parameters
        ----------
        path : str
            Path to the file to save model to.
        """
        if self.estimator is not None:
            with open(path, "wb") as file:
                dill.dump(self.estimator, file)
        else:
            raise ValueError("HMM estimator does not exist. Check your cofig for 'estimator'.")

    def load(self, path, *args, **kwargs):  # pylint: disable=arguments-differ
        """Load ``HMModel`` from file with ``dill``.

        Parameters
        ----------
        path : str
            Path to the model.
        """
        with open(path, "rb") as file:
            self.estimator = dill.load(file)
            
            
    def _make_prediction_inputs(self, *args, targets=None, feed_dict=None, **kwargs):
        """ Parse arguments to create valid inputs for the model.
        Implements the logic of parsing the positional and keyword arguments to the model,
        possibly wrapped into `feed_dict` dictionary, or even combination of the two.
        Used under the hood of :meth:`~.TorchModel.predict` method.
        Examples
        --------
        .. code-block:: python
            model.predict(B('images'), targets=B('labels'))
            model.predict(images=B('images'), targets=B('labels'))
            model.predict(B('images'), targets=B('labels'), masks=B('masks'))
        """
        # Concatenate `kwargs` and `feed_dict`; if not empty, use keywords in `parse_input`
        feed_dict = {**(feed_dict or {}), **kwargs}
        if len(feed_dict) == 1:
            _, value = feed_dict.popitem()
            args = (*args, value)
        if feed_dict:
            if targets is not None and 'targets' in feed_dict.keys():
                warnings.warn("`targets` already present in `feed_dict`, so those passed as keyword arg won't be used")
            *inputs, targets = self.parse_inputs(*args, **feed_dict)

        # Positional arguments only
        else:
            inputs = self.parse_inputs(*args)
            if targets is not None:
                targets = self.parse_inputs(targets)[0]
        inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs
        return inputs, targets
 
    def train(self, feed_dict=None, *args, **kwargs):
        """ Train the model using data provided.

        Parameters
        ----------
        X : array-like
            A matrix of observations.
            Should be of shape (n_samples, n_features).
        lengths : array-like of integers optional
            If present, should be of shape (n_sequences, ).
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Notes
        -----
        For more details and other parameters look at the documentation for the estimator used.
        """
        
        X, lengths = self._make_prediction_inputs(*args, targets=targets, feed_dict=feed_dict, **kwargs)
        print(X.shape)
        print(X)
        
        self.estimator.fit(X, lengths)
        return list(self.estimator.monitor_.history)

    def predict(self, feed_dict=None, *args, **kwargs):
        """ Make prediction with the data provided.

        Parameters
        ----------
        X : array-like
            A matrix of observations.
            Should be of shape (n_samples, n_features).
        lengths : array-like of integers optional
            If present, should be of shape (n_sequences, ).
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        output: array
            Labels for each sample of X.

        Notes
        -----
        For more details and other parameters look at the documentation for the estimator used.
        """
        X, lengths = self._make_prediction_inputs(*args, targets=targets, feed_dict=feed_dict, **kwargs)
        preds = self.estimator.predict(X, lengths)
        if lengths:
            output = np.array(np.split(preds, np.cumsum(lengths)[:-1]) + [None])[:-1]
        else:
            output = preds
        return output
