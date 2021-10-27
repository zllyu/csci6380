import numpy as np


def letter_to_index(letter, type):
    """
        Covert English letters to indices of the letters.
        :param letter: one English letter.
        :param type: which class the letter belongs to.
        :return: index number.
    """
    z_values = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
    p_values = ['X', 'R', 'S', 'A', 'H', 'K']
    c_values = ['X', 'O', 'I', 'C']
    if type == 'z':
        index = z_values.index(letter)
    elif type == 'p':
        index = p_values.index(letter)
    elif type == 'c':
        index = c_values.index(letter)
    else:
        index = -1
    return index


def digit_to_onehot(origin_matrix, class_number):
    """
        Covert a matrix using one-hot encoding.
        :param origin_matrix: a numpy array with shape [value_count].
        :param class_number: the number of classes.
        :return: one hot encoded matrix with shape [value_count, class_number].
    """
    eye = np.eye(class_number, dtype=int)
    encoding_matrix = eye[origin_matrix]
    return encoding_matrix


def transform_data(dataset):
    """
        Transform DataFrame to feature numpy matrix and target numpy matrix.
        :param dataset: type-> pandas.DataFrame.
        :return: feature numpy matrix, target numpy matrix.
    """
    dataset['z-values'] = dataset['z-values'].apply(letter_to_index, args='z')
    dataset['p-values'] = dataset['p-values'].apply(letter_to_index, args='p')
    dataset['c-values'] = dataset['c-values'].apply(letter_to_index, args='c')
    z_onehot = digit_to_onehot(np.array(dataset['z-values']), class_number=7)
    p_onehot = digit_to_onehot(np.array(dataset['p-values']), class_number=6)
    c_onehot = digit_to_onehot(np.array(dataset['c-values']), class_number=4)
    other_class = np.array(dataset.loc[:, 'activity':'largest spot area'])
    X_np = np.concatenate((z_onehot, p_onehot, c_onehot, other_class), axis=-1)
    Y_np = np.array(dataset.loc[:, 'C-class':])
    return X_np, Y_np
