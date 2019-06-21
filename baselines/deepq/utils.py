from baselines.common.input import observation_input
from baselines.common.tf_util import adjust_shape
import tensorflow as tf

# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplementedError

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplementedError


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: adjust_shape(self._placeholder, data)}


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """Creates an input placeholder tailored to a specific observation space

        Parameters
        ----------

        observation_space:
                observation space of the environment. Should be one of the gym.spaces types
        name: str
                tensorflow name of the underlying placeholder
        """
        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt


def cat_entropy_softmax(p0):
    min_el = tf.reduce_min(p0)
    partial_result_0 = p0
    # zero = tf.constant(0.0)
    print('min_el.shape:', min_el.shape)
    me = min_el[0]
    print('me.shape:', me.shape)

    if me < 0:
        temp_tensor = tf.fill(partial_result_0.shape, - me)
        partial_result_0 = tf.add(partial_result_0, temp_tensor)
    partial_result_1 = tf.log(partial_result_0 + 1e-6)
    partial_result_2 = p0 * partial_result_1
    result = - tf.reduce_sum(partial_result_2, axis=1)
    return partial_result_0, partial_result_1, partial_result_2, result
