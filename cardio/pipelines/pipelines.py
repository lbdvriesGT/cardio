"""Contains pipelines."""

from functools import partial
import numpy as np

import tensorflow as tf
from hmmlearn import hmm

import cardio.dataset as ds
from batchflow import F, V, B
from ..models.hmm import HMModel, prepare_hmm_input



def hmm_preprocessing_pipeline(batch_size=20, features="hmm_features"):
    """Preprocessing pipeline for Hidden Markov Model.

    This pipeline prepares data for ``hmm_train_pipeline``.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    batch_size : int
        Number of samples in batch.
        Default value is 20.
    features : str
        Batch attribute to store calculated features.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    def get_annsamples(batch):
        """Get annsamples from annotation
        """
        return [ann["annsamp"] for ann in batch.annotation]

    def get_anntypes(batch):
        """Get anntypes from annotation
        """
        return [ann["anntype"] for ann in batch.annotation]

    return (ds.Pipeline()
      .init_variable("annsamps", list)
      .init_variable("anntypes", list)
      .init_variable("hmm_features", list)
      .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
      .cwt(src="signal", dst="hmm_features", scales=[4,8,16], wavelet="mexh")
      .standardize(axis=-1, src="hmm_features", dst="hmm_features")
      .update(V("annsamps"), F(get_annsamples, mode='e')(batch=B()))
      .update(V("anntypes"), F(get_anntypes, mode='e')(batch=B()))
      .update(V("hmm_features"), ds.B("hmm_features"))
      .run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True, bar=True))

def hmm_train_pipeline(hmm_preprocessed, batch_size=20, features="hmm_features", channel_ix=0,
                       n_iter=25, random_state=42, model_name='HMM'):
    """Train pipeline for Hidden Markov Model.

    This pipeline trains hmm model to isolate QRS, PQ and QT segments.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    hmm_preprocessed : Pipeline
        Pipeline with precomputed hmm features through ``hmm_preprocessing_pipeline``
    batch_size : int
        Number of samples in batch.
        Default value is 20.
    features : str
        Batch attribute to store calculated features.
    channel_ix : int
        Index of signal's channel, which should be used in training and predicting.
    n_iter : int
        Number of learning iterations for ``HMModel``.
    random_state: int
        Random state for ``HMModel``.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    def prepare_means_covars(hmm_features, clustering, states=(3, 5, 11, 14, 17, 19), num_states=19, num_features=3):
        """This function is specific to the task and the model configuration, thus contains hardcode.
        """
        means = np.zeros((num_states, num_features))
        covariances = np.zeros((num_states, num_features, num_features))

        # Prepearing means and variances
        last_state = 0
        unique_clusters = len(np.unique(clustering)) - 1 # Excuding value -1, which represents undefined state
        for state, cluster in zip(states, np.arange(unique_clusters)):
            value = hmm_features[clustering == cluster, :]
            means[last_state:state, :] = np.mean(value, axis=0)
            covariances[last_state:state, :, :] = value.T.dot(value) / np.sum(clustering == cluster)
            last_state = state

        return means, covariances

    def prepare_transmat_startprob():
        """ This function is specific to the task and the model configuration, thus contains hardcode.
        """
        # Transition matrix - each row should add up tp 1
        transition_matrix = np.diag(19 * [14/15.0]) + np.diagflat(18 * [1/15.0], 1) + np.diagflat([1/15.0], -18)

        # We suppose that absence of P-peaks is possible
        transition_matrix[13, 14] = 0.9*1/15.0
        transition_matrix[13, 17] = 0.1*1/15.0

        # Initial distribution - should add up to 1
        start_probabilities = np.array(19 * [1/np.float(19)])

        return transition_matrix, start_probabilities

    def expand_annotation(annsamp, anntype, length):
        """Unravel annotation
        """
        begin = -1
        end = -1
        s = 'none'
        states = {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
        annot_expand = -1 * np.ones(length)

        for j, samp in enumerate(annsamp):
            if anntype[j] == '(':
                begin = samp
                if (end > 0) & (s != 'none'):
                    if s == 'N':
                        annot_expand[end:begin] = states['st']
                    elif s == 't':
                        annot_expand[end:begin] = states['iso']
                    elif s == 'p':
                        annot_expand[end:begin] = states['pq']
            elif anntype[j] == ')':
                end = samp
                if (begin > 0) & (s != 'none'):
                    annot_expand[begin:end] = states[s]
            else:
                s = anntype[j]

        return annot_expand

    lengths = [features_iter.shape[2] for features_iter in hmm_preprocessed.get_variable(features)]
    hmm_features = np.concatenate([features_iter[channel_ix, :, :].T for features_iter
                                   in hmm_preprocessed.get_variable(features)])
    anntype = hmm_preprocessed.get_variable("anntypes")
    annsamp = hmm_preprocessed.get_variable("annsamps")

    expanded = np.concatenate([expand_annotation(samp, types, length) for
                               samp, types, length in zip(annsamp, anntype, lengths)])
    means, covariances = prepare_means_covars(hmm_features, expanded, states=[3, 5, 11, 14, 17, 19], num_features=3)
    transition_matrix, start_probabilities = prepare_transmat_startprob()

    config_train = {
        'build': True,
        'estimator': hmm.GaussianHMM(n_components=19, n_iter=n_iter, covariance_type="full", random_state=random_state,
                                     init_params='', verbose=False),
        'init_params': {'means_': means, 'covars_': covariances, 'transmat_': transition_matrix,
                        'startprob_': start_probabilities}
    }

    return (ds.Pipeline()
            .init_model(model_name, HMModel, "dynamic", config=config_train)
            .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
            .cwt(src="signal", dst=features, scales=[4, 8, 16], wavelet="mexh")
            .standardize(axis=-1, src=features, dst=features)
            .train_model(model_name, make_data=partial(prepare_hmm_input, features=features, channel_ix=channel_ix))
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True, bar=True))

def hmm_predict_pipeline(model_path, batch_size=20, features="hmm_features",
                         channel_ix=0, annot="hmm_annotation", model_name='HMM'):
    """Prediction pipeline for Hidden Markov Model.

    This pipeline isolates QRS, PQ and QT segments.
    It works with dataset that generates batches of class ``EcgBatch``.

    Parameters
    ----------
    model_path : str
        Path to pretrained ``HMModel``.
    batch_size : int
        Number of samples in batch.
        Default value is 20.
    features : str
        Batch attribute to store calculated features.
    channel_ix : int
        Index of channel, which data should be used in training and predicting.
    annot: str
        Specifies attribute of batch in which annotation will be stored.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    config_predict = {
        'build': False,
        'load': {'path': model_path}
    }

    return (ds.Pipeline()
            .init_model(model_name, HMModel,"static", config=config_predict)
            .load(fmt="wfdb", components=["signal", "meta"])
            .cwt(src="signal", dst=features, scales=[4, 8, 16], wavelet="mexh")
            .standardize(axis=-1, src=features, dst=features)
            .predict_model(model_name, make_data=partial(prepare_hmm_input, features=features, channel_ix=channel_ix),
                           save_to=ds.B(annot), mode='w')
            .calc_ecg_parameters(src=annot)
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True, bar=True))
