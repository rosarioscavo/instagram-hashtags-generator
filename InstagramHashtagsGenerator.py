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
k_means = MiniBatchKMeans(NUM_VOCABULARY)
warnings.filterwarnings("ignore")


# it shows an image in the default image viewer
def show_image(image, tag):
    plt.title(f"Tag: {tag}")
    plt.axis('off')
    plt.imshow(image)
    plt.show()


# method to show the content of a DataFrame extending the columns
def print_data(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data)


# It removes some common tags that are considered useless
def clean_tags(tags):
    stop_words = ['love', 'photography', 'instagood', 'travel', 'beautiful', 'style', 'follow', 'photooftheday',
                  'picoftheday', 'instagram', 'photo', 'naturephotography', 'life',
                  'instadaily', 'travelphotography']
    tags_cleaned = list(filter(lambda tag: tag not in stop_words, tags))
    return tags_cleaned


# starting from a path to the data_set folder that contains some folders,
# create a CSV with 3 columns(the path of the image, class and relative list of tags)
def csv_creation(data_set_path, main_tags_path):
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


# return a DataFrame filled by a CSV, located in the path
def get_data_csv(path):
    data = pd.read_csv(path)
    return data


# return a list of tags as a Series object
def get_tags_from_csv(path_csv):
    data = get_data_csv(path_csv)
    return data['tags']


# return a dictionary with the frequency of each tag in the tags Series. Each element of tags is a string of tags.
# Using literal_eval it is possible to convert a string into a list obtaining a list of tags
def tags_frequency(tags, num_tags):
    # Creating a list of tags using the list comprehension
    list_tags = [tag for index in range(0, len(tags)) for tag in literal_eval(tags[index])]
    counter = Counter(list_tags)
    most_common_counter = counter.most_common(num_tags)
    most_common_tags = [tag for tag, value in most_common_counter]
    return most_common_tags


# It creates a file containing each tag in tags, one per line
def create_file_most_common_tags(path, tags):
    with open(path, "w") as output:
        for tag in tags:
            output.write(str(tag) + '\n')


# It saves a .pkl file
def save_object(path, data):
    pickle.dump(data, open(path + '.pkl', "wb"))


# It loads a .pkl file
def load_object(path):
    data = pickle.load(open(path + '.pkl', "rb"))
    return data


# It creates a set for a tag in the file path_tags. The set's balance depends on the balance factor
def set_creation(data, tag, balance=1):
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


# creating a training and testing sets for each tag in the file path_tags
def training_testing_set_creation(path_tags):
    data = get_data_csv(DATASET_NAME_CSV)
    target_flags = open(path_tags).read().splitlines()

    for tag in target_flags:
        training_set = set_creation(data, tag, balance=1)
        save_object(TRAINING_SET_FOLDER + os.path.sep + tag, training_set)

        testing_set = set_creation(data, tag, balance=3)
        save_object(TESTING_SET_FOLDER + os.path.sep + tag, testing_set)
        print(tag + ' completed')


# it shows a pie with the percentage of each class in the data set
def show_data_balance(data):
    data.groupby('class')['class'].count().plot.pie(autopct='%.2f', subplots=True)
    plt.show()


# it extracts and describes all the patches of the images in data using the dsift function
def extract_and_describe(data, size=5, step=STEP):
    descriptors = []
    for i, row in tqdm(data.iterrows(), "Extracting/Describing Patches", total=len(data)):
        path = DATA_SET_FOLDER + os.path.sep + row['path']
        im = io.imread(path, as_gray=True)

        _, description = dsift(im, size=size, step=step, fast=True)
        descriptors.append(description)

    return np.vstack(descriptors)


# it describes an image and then return the bag of visual words
def load_and_describe(filename, size=5, step=STEP):
    im = io.imread(filename, as_gray=True)
    _, descriptors = dsift(im, size=size, step=step, fast=True)
    tokens = k_means.predict(descriptors)
    return tokens


# it creates and saves the bag of visual words for train and test sets applying tf-idf normalization
def bovw_normalized_creation(train_descriptions, tag, train_set, test_set):
    # k_means needs to be fitted before executing load_and_describe function
    k_means.fit(train_descriptions)
    save_object(K_MEANS_FOLDER + os.path.sep + tag + '_kmeans', k_means)

    tfidf = TfidfVectorizer(tokenizer=load_and_describe, vocabulary=range(NUM_VOCABULARY), use_idf=True)

    x_train = tfidf.fit_transform(DATA_SET_FOLDER + os.path.sep + train_set['path'])
    save_object(BOVW_FOLDER + os.path.sep + tag + '_train', x_train)

    x_test = tfidf.transform(DATA_SET_FOLDER + os.path.sep + test_set['path'])
    save_object(BOVW_FOLDER + os.path.sep + tag + '_test', x_test)

    save_object(TFIDF_FOLDER + os.path.sep + tag, tfidf)


# fitting process for Nearest Neighbor
def nearest_neighbor_fitting(x_train, y_train, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn


# it calculates the best hyperparameter for K-nearest neighbors classifier
def hyperparameter_optimization_knn(x_valid, y_valid):
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


# fitting process for Logistic Regression
def logistic_regression_fitting(x_train, y_train):
    lr = LogisticRegression(multi_class='ovr', solver='sag')
    lr.fit(x_train, y_train)
    return lr


# classifying the dataset(train and test set)
def classification_process(classifier, x_train, x_test):
    y_train_pred = classifier.predict(x_train)
    y_test_pred = classifier.predict(x_test)

    return y_train_pred, y_test_pred


# method to create all the files for the classifiers, for each tag
def classifiers_creation(path_tags):
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


# it creates a plot showing the performance measures of each classifier
def show_performance_measures(path_tags):
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


# it applies the binary classifier(relative to the tag) to the image
def image_classification(path_image, tag, classifier):
    tag_pred = None
    tfidf = load_object(TFIDF_FOLDER + os.path.sep + tag)
    feats = tfidf.transform(path_image)
    pred = classifier.predict(feats)
    pred = pred[0]
    if 'not' not in pred:
        tag_pred = pred

    return tag_pred


# it applies all the binary classifiers and returns a list of tags. It possible to apply at the same time lr and knn.
# the classifier parameter can take the following value: 'lr', 'knn' and 'lr + knn', default value: 'lr + knn'
def instagram_hashtags_generator(path_image, path_tags, classifier='lr + knn'):
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
        print("lr: ",  ', '.join(lr_tags))
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

    #knn, lr, lr + knn
    classifier = 'lr + knn'
    path_image='dataset/paint.jpg'
    instagram_hashtags_generator(path_image, LIST_COMMON_TAGS_NAME, classifier)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
