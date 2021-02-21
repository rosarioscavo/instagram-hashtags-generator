import json
import re
import os
import pickle
import pandas as pd
import numpy as np
import warnings
import time
from ast import literal_eval
from collections import Counter

from matplotlib import pyplot as plt
from skimage import io
from cyvlfeat.sift import dsift
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATASET_NAME_CSV = 'resource/dataset.csv'
LIST_MAIN_TAGS_NAME = 'resource/list tags.txt'
LIST_COMMON_TAGS_NAME = 'resource/common tags.txt'
TESTING_SET_FOLDER = 'resource/Testing sets'
TRAINING_SET_FOLDER = 'resource/Training sets'
DATA_SET_FOLDER = 'dataset'
TRAINING_DESCRIPTORS = 'resource/Training descriptors'
K_MEANS_FOLDER = 'resource/KMeans'
BOVW_FOLDER = 'resource/BOVW'
TFIDF_FOLDER = 'resource/TFIDF'
LR_FOLDER = 'resource/LR'
KNN_FOLDER = 'resource/KNN'
NUM_TAGS = 20
STEP = 60
NUM_VOCABULARY = 500

global k_means
# noinspection PyRedeclaration
k_means = MiniBatchKMeans(NUM_VOCABULARY)
warnings.filterwarnings("ignore")


def show_image(image, tag):
    """
    Show an image in the default image viewer.

    Parameters
    ----------
    image : str
        Path to the image.
    tag : str
        Tag of the image.

    Returns
    -------
    None
    """
    plt.title(f"Tag: {tag}")
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def print_data(data):
    """
    Show the content of a DataFrame extending the columns.

    Parameters
    ----------
    data : Dataframe
        The dataframe to be shown.

    Returns
    -------
    None
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data)


def clean_tags(tags):
    """
    Remove some common tags that are considered useless.

    Parameters
    ----------
    tags : List
        List of tags.
    Returns
    -------
    list
        List of tags cleaned.
    """
    stop_words = ['love', 'photography', 'instagood', 'travel', 'beautiful', 'style', 'follow', 'photooftheday',
                  'picoftheday', 'instagram', 'photo', 'naturephotography', 'life',
                  'instadaily', 'travelphotography']
    tags_cleaned = list(filter(lambda tag: tag not in stop_words, tags))
    return tags_cleaned


def csv_creation(data_set_path, main_tags_path):
    """
    Create a CSV with 3 columns(the path of the image, class and relative list of tags).

    Parameters
    ----------
    data_set_path : str
        Path to the dataset.
    main_tags_path : str
        List to the main tags.

    Returns
    -------
    None
    """
    df = pd.DataFrame(columns=['path', 'class', 'tags'])
    target_flags = open(main_tags_path).read().splitlines()

    for tag in target_flags:
        path = data_set_path + os.path.sep + tag + os.path.sep + tag + '.json'
        count = 0

        with open(path) as json_file:
            data = json.load(json_file)
            for p in data['GraphImages']:
                if count >= 1000:
                    break

                if len(p['urls']) == 0:
                    continue

                url = p['urls'][0]
                result = re.search('/[0-9](.*)\.jpg', url)
                if result is None:
                    continue

                url_final = tag + result.group(0)

                try:
                    tags = p['tags']
                except KeyError:
                    # this tuple doesn't contain any 'tags' attribute
                    continue

                if len(tags) == 0:
                    tags = [tag]

                tags = clean_tags(tags)

                new_row = {'path': url_final, 'class': tag, 'tags': tags}
                df = df.append(new_row, ignore_index=True)
                count += 1

        print('\'' + tag + '\'' + " tag completed")

    # index=False because I don't need to save a column with indexes
    df.to_csv(DATASET_NAME_CSV, index=False)


def get_data_csv(path):
    """
    Return a DataFrame filled by a CSV, located in the path.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe with the content of the file
    """
    data = pd.read_csv(path)
    return data


def get_tags_from_csv(path_csv):
    """
    Return a list of tags as a Series object.

    Parameters
    ----------
    path_csv : str
        Path to a list of tags.

    Returns
    -------
    pandas.core.series.Series
        List of tags.
    """
    data = get_data_csv(path_csv)
    return data['tags']


def tags_frequency(tags, num_tags):
    """
    Return a list of the most common tags.

    Parameters
    ----------
    tags : pandas.core.series.Series
        List of strings of tags.
    num_tags : int
        The number of most common tags to select.

    Returns
    -------
    list
        List of the most common tags.
    """
    # Creating a list of tags using the list comprehension
    # Using literal_eval it is possible to convert a string into a list obtaining a list of tags
    list_tags = [tag for index in range(0, len(tags)) for tag in literal_eval(tags[index])]
    counter = Counter(list_tags)
    most_common_counter = counter.most_common(num_tags)
    most_common_tags = [tag for tag, value in most_common_counter]
    return most_common_tags


def create_file_most_common_tags(path, tags):
    """
    Create a file containing each tag in tags, one per line.

    Parameters
    ----------
    path : str
        Path to save the file.
    tags : list
        List of tags.

    Returns
    -------
    None
    """
    with open(path, "w") as output:
        for tag in tags:
            output.write(str(tag) + '\n')


def save_object(path, data):
    """
    Save a .pkl file.

    Parameters
    ----------
    path : str
        Path to save the file.
    data : Any
        Data to be saved.

    Returns
    -------
    None
    """
    pickle.dump(data, open(path + '.pkl', "wb"))


def load_object(path):
    """
    Load a .pkl file.

    Parameters
    ----------
    path : str
        Path of the .pkl file to be loaded.

    Returns
    -------
    Any
        Data of the .pkl file loaded.
    """
    data = pickle.load(open(path + '.pkl', "rb"))
    return data


#
def set_creation(data, tag, balance=1):
    """
    Create a Dataset for a tag. The set's balance depends on the balance factor.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Dataframe with the content of the file.
    tag : str
        The tag.
    balance : int, optional
        The balance factor.
        It sets the percentage of the 'positive' and 'negative' classes.

    Returns
    -------
    pandas.core.frame.DataFrame
        The dataset of the specific tag.
    """
    data_set = pd.DataFrame()

    # it shuffles the dataframe
    temp_data = data.copy()
    data_shuffle = temp_data.sample(frac=1)

    for i in range(0, len(temp_data)):
        if tag in temp_data.iloc[i]['tags']:
            new_row = temp_data.iloc[i].copy()
            new_row['class'] = tag
            data_set = data_set.append(new_row)

    num_rows = len(data_set)
    counter = 0

    for i in range(0, len(data_shuffle)):
        if counter >= num_rows * balance:
            break
        if tag not in data_shuffle.iloc[i]['tags']:
            new_row = data_shuffle.iloc[i].copy()
            new_row['class'] = 'not ' + tag
            data_set = data_set.append(new_row)
            counter += 1

    return data_set


def training_testing_set_creation(path_tags):
    """
    Create a training and testing sets for each tag in the file path_tags.

    Parameters
    ----------
    path_tags : str
        Path to the file containing tags.

    Returns
    -------
    None
    """
    data = get_data_csv(DATASET_NAME_CSV)
    target_flags = open(path_tags).read().splitlines()

    for tag in target_flags:
        training_set = set_creation(data, tag, balance=1)
        save_object(TRAINING_SET_FOLDER + os.path.sep + tag, training_set)

        testing_set = set_creation(data, tag, balance=3)
        save_object(TESTING_SET_FOLDER + os.path.sep + tag, testing_set)
        print(tag + ' completed')


def show_data_balance(data):
    """
    Show a pie with the percentage of each class in the dataset.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        The dataset.

    Returns
    -------
    None
    """
    data.groupby('class')['class'].count().plot.pie(autopct='%.2f', subplots=True)
    plt.show()


def extract_and_describe(data, size=5, step=STEP):
    """
    Extract and describe all the patches of the images in data using the dsift function.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        The dataset.
    size : int
         The size of the spatial bin of the SIFT descriptor in pixels.
    step : int
         A SIFT descriptor is extracted every ``step`` pixels.

    Returns
    -------
    np.ndarray
        Features of the dataset's images.
    """
    descriptors = []
    for i, row in tqdm(data.iterrows(), "Extracting/Describing Patches", total=len(data)):
        path = DATA_SET_FOLDER + os.path.sep + row['path']
        im = io.imread(path, as_gray=True)

        _, description = dsift(im, size=size, step=step, fast=True)
        descriptors.append(description)

    return np.vstack(descriptors)


def load_and_describe(filename, size=5, step=STEP):
    """
    Describe an image and then return the bag of visual words.

    Parameters
    ----------
    filename : str
        Path to an image.
    size : int
         The size of the spatial bin of the SIFT descriptor in pixels.
    step : int
         A SIFT descriptor is extracted every ``step`` pixels.

    Returns
    -------
    list
        List of the closest cluster for each descriptor.
    """
    im = io.imread(filename, as_gray=True)
    _, descriptors = dsift(im, size=size, step=step, fast=True)
    tokens = k_means.predict(descriptors)
    return tokens


def bovw_normalized_creation(train_descriptions, tag, train_set, test_set):
    """
    Create and save the bag of visual words for train and test sets applying the tf-idf normalization.

    Parameters
    ----------
    train_descriptions : np.ndarray
        Features of the dataset's images.
    tag : str
        The tag of the dataset's.
    train_set : pandas.core.frame.DataFrame
        The training set.
    test_set : pandas.core.frame.DataFrame
        The testing set.

    Returns
    -------
    None
    """
    # k_means needs to be fitted before executing load_and_describe function
    k_means.fit(train_descriptions)
    save_object(K_MEANS_FOLDER + os.path.sep + tag + '_kmeans', k_means)

    tfidf = TfidfVectorizer(tokenizer=load_and_describe, vocabulary=range(NUM_VOCABULARY), use_idf=True)

    x_train = tfidf.fit_transform(DATA_SET_FOLDER + os.path.sep + train_set['path'])
    save_object(BOVW_FOLDER + os.path.sep + tag + '_train', x_train)

    x_test = tfidf.transform(DATA_SET_FOLDER + os.path.sep + test_set['path'])
    save_object(BOVW_FOLDER + os.path.sep + tag + '_test', x_test)

    save_object(TFIDF_FOLDER + os.path.sep + tag, tfidf)


def nearest_neighbor_fitting(x_train, y_train, k=1):
    """
    Fit process for Nearest Neighbor.

    Parameters
    ----------
    x_train : pandas.core.series.Series
        BoVWs of the training set.
    y_train : pandas.core.series.Series
        Classes of the training set.
    k : int
        Number of neighbors.

    Returns
    -------
    KNeighborsClassifier
        The fitted KNN model.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn


def hyperparameter_optimization_knn(x_valid, y_valid):
    """
    Calculate the best hyperparameter for K-nearest neighbors classifier.

    Parameters
    ----------
    x_valid : pandas.core.series.Series
        BoVWs of the validation set
    y_valid : pandas.core.series.Series
        Classes label of the validation set.

    Returns
    -------
    int
        The best hyperparameter for K-nearest neighbors classifier.
    """
    best_k = 1
    best_score = 0
    for k in range(1, 10):
        knn = nearest_neighbor_fitting(x_valid, y_valid, k)
        y_valid_pred = knn.predict(x_valid)
        score = f1_score(y_valid, y_valid_pred, average=None).mean()
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def logistic_regression_fitting(x_train, y_train):
    """
    Fit process for Logistic Regression.

    Parameters
    ----------
    x_train : pandas.core.series.Series
        BoVWs of the training set.
    y_train : pandas.core.series.Series
        Classes label of the training set.

    Returns
    -------
    LogisticRegression
        The fitted Logistic Regression model.
    """
    lr = LogisticRegression(multi_class='ovr', solver='sag')
    lr.fit(x_train, y_train)
    return lr


def classification_process(classifier, x_train, x_test):
    """
    Classifies the dataset (train and test set).

    Parameters
    ----------
    classifier : KNeighborsClassifier, LogisticRegression
        The model of the classifier (knn or lr).
    x_train : pandas.core.series.Series
        BoVWs of the training set.
    x_test : pandas.core.series.Series
        BoVWs of the testing set.

    Returns
    -------
    tuple
        Return the predicted classes of the training and testing sets.

    """
    y_train_pred = classifier.predict(x_train)
    y_test_pred = classifier.predict(x_test)

    return y_train_pred, y_test_pred


def classifiers_creation(path_tags):
    """
    Method to create all the files for the classifiers for each tag.
    Parameters
    ----------
    path_tags : str
        Path to the file containing the tags of each classifier.

    Returns
    -------
    None
    """
    target_flags = open(path_tags).read().splitlines()

    for tag in target_flags:
        train_set = load_object(TRAINING_SET_FOLDER + os.path.sep + tag)
        test_set = load_object(TESTING_SET_FOLDER + os.path.sep + tag)
        y_train = train_set['class']
        y_test = test_set['class']

        train_descriptions = extract_and_describe(train_set)
        save_object(TRAINING_DESCRIPTORS + os.path.sep + tag, train_descriptions)

        print("train_description for tag: ", tag, " completed")
        bovw_normalized_creation(train_descriptions, tag, train_set, test_set)
        print("bovw for tag: ", tag, " completed")
        x_train = load_object(BOVW_FOLDER + os.path.sep + tag + '_train')
        x_test = load_object(BOVW_FOLDER + os.path.sep + tag + '_test')

        lr = logistic_regression_fitting(x_train, y_train)
        save_object(LR_FOLDER + os.path.sep + tag, lr)

        # creating the validation set
        x_train, x_valid, y_train, y_valid = train_test_split(x_test, y_test, test_size=0.30)

        best_k = hyperparameter_optimization_knn(x_valid, y_valid)
        knn = nearest_neighbor_fitting(x_train, y_train, best_k)
        save_object(KNN_FOLDER + os.path.sep + tag, knn)
        print("tag: ", tag, " completed")


def show_performance_measures(path_tags):
    """
    It creates a plot showing the performance measures of each classifier.

    Parameters
    ----------
    path_tags : str
        Path to the file containing the tags of each classifier.

    Returns
    -------
    None
    """
    target_flags = open(path_tags).read().splitlines()
    measures = pd.DataFrame(columns=['Tag'])
    measures = measures.set_index('Tag')

    for tag in target_flags:
        test_set = load_object(TESTING_SET_FOLDER + os.path.sep + tag)
        x_train = load_object(BOVW_FOLDER + os.path.sep + tag + '_train')
        x_test = load_object(BOVW_FOLDER + os.path.sep + tag + '_test')
        y_test = test_set['class']

        lr = load_object(LR_FOLDER + os.path.sep + tag)
        knn = load_object(KNN_FOLDER + os.path.sep + tag)

        _, y_test_pred = classification_process(lr, x_train, x_test)
        lr_score = f1_score(y_test, y_test_pred, average=None).mean()

        _, y_test_pred = classification_process(knn, x_train, x_test)
        knn_score = f1_score(y_test, y_test_pred, average=None).mean()

        new_score = pd.Series({'F1 LR': lr_score, 'F1 KNN': knn_score}, name=tag)
        measures = measures.append(new_score)

    measures.plot.barh()
    plt.grid()
    plt.show()


def image_classification(path_image, tag, classifier):
    """
    It applies the binary classifier (relative to the tag) to an image.

    Parameters
    ----------
    path_image : str
        Path to the image.
    tag : str
        Tag for which the classification is going to be performed.
    classifier : KNeighborsClassifier, LogisticRegression
        The classifier (knn or lr) relative to the tag.

    Returns
    -------
    str
        The relative tag or None, depending on the classification process.
    """
    tag_pred = None
    tfidf = load_object(TFIDF_FOLDER + os.path.sep + tag)
    feats = tfidf.transform(path_image)
    pred = classifier.predict(feats)
    pred = pred[0]
    if 'not' not in pred:
        tag_pred = pred

    return tag_pred


def instagram_hashtags_generator(path_image, path_tags, classifier='lr + knn'):
    """
    It applies all the binary classifiers to an image and prints a list of tags. It possible to apply at the same time lr and knn.

    Parameters
    ----------
    path_image : str
        Path to the image.

    path_tags : str
        Path to the tags that will be analyzed.

    classifier : str, optional
        The classifiers that should be performed (the default value is 'lr + knn').
        The accepted values are: 'lr', 'knn' and 'lr + knn'.

    Returns
    -------
    """

    global k_means
    path_image = [path_image]
    knn_tags = []
    lr_tags = []
    target_flags = open(path_tags).read().splitlines()

    for tag in target_flags:
        k_means = load_object(K_MEANS_FOLDER + os.path.sep + tag + '_kmeans')
        if classifier == 'lr' or classifier == 'lr + knn':
            lr = load_object(LR_FOLDER + os.path.sep + tag)
            tag_pred = image_classification(path_image, tag, lr)
            if tag_pred:
                lr_tags.append(tag_pred)
        if classifier == 'knn' or classifier == 'lr + knn':
            knn = load_object(KNN_FOLDER + os.path.sep + tag)
            tag_pred = image_classification(path_image, tag, knn)
            if tag_pred:
                knn_tags.append(tag_pred)

    if classifier == 'lr' or classifier == 'lr + knn':
        print("lr: ", ', '.join(lr_tags))
    if classifier == 'knn' or classifier == 'lr + knn':
        print("knn: ", ', '.join(knn_tags))


# main
if __name__ == '__main__':
    start = time.time()

    # csv_creation(DATA_SET_FOLDER, LIST_MAIN_TAGS_NAME)
    # tags_csv = get_tags_from_csv(DATASET_NAME_CSV)
    # common_tags = tags_frequency(tags_csv, NUM_TAGS)
    # create_file_most_common_tags(LIST_COMMON_TAGS_NAME, common_tags)
    # training_testing_set_creation(LIST_COMMON_TAGS_NAME)
    # classifiers_creation(LIST_COMMON_TAGS_NAME)
    # show_performance_measures(LIST_COMMON_TAGS_NAME)

    # knn, lr, lr + knn
    classifier = 'lr + knn'
    path_image = 'images/paint.jpg'
    instagram_hashtags_generator(path_image, LIST_COMMON_TAGS_NAME, classifier)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
