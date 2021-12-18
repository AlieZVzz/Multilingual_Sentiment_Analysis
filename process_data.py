from Config import *
import numpy as np


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def build_word2id(train_path, validation_path):
    word2id = {'PAD': 0}
    paths = [train_path, validation_path]
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    tag2id = {'0': 0, '1': 1}

    return word2id, tag2id


def load_data(train_path, validation_path, test_path, word2id, tag2id):
    x_train, y_train = [], []

    x_validation, y_validation = [], []

    x_test, y_test = [], []

    x_train_id, x_validation_id, x_test_id = [], [], []
    y_train_id, y_validation_id, y_test_id = [], [], []

    with open(train_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_train.append(data[0])
            x_train.append(data[1].strip().split())

    with open(validation_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_validation.append(data[0])
            x_validation.append(data[1].strip().split())

    with open(test_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_test.append(data[0])
            x_test.append(data[1].strip().split())

    for i in range(len(x_train)):
        x_train_id.append([word2id[x] for x in x_train[i] if x in word2id])
        # y_train_id.append([tag2id[x] for x in y_train[i] if x in tag2id])
        y_train_id += [tag2id[x] for x in y_train[i] if x in tag2id]

    for i in range(len(x_validation)):
        x_validation_id.append([word2id[x] for x in x_validation[i] if x in word2id])
        # y_validation_id.append([tag2id[x] for x in y_validation[i] if x in tag2id])
        y_validation_id += [tag2id[x] for x in y_validation[i] if x in tag2id]

    for i in range(len(x_test)):
        x_test_id.append([word2id[x] for x in x_test[i] if x in word2id])
        # y_test_id.append([tag2id[x] for x in y_test[i] if x in tag2id])
        y_test_id += [tag2id[x] for x in y_test[i] if x in tag2id]

    return x_train_id, y_train_id, x_validation_id, y_validation_id, x_test_id, y_test_id


def process_data(out):
    x_train = pad_sequences(out[0], maxlen=60, padding='post', value=0)
    y_train = out[1]

    x_validation = pad_sequences(out[2], maxlen=60, padding='post', value=0)
    y_validation = out[3]

    x_test = pad_sequences(out[4], maxlen=60, padding='post', value=0)
    y_test = out[5]

    return x_train, y_train, x_validation, y_validation, x_test, y_test
