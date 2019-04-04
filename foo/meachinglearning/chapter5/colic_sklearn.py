from sklearn.linear_model import LogisticRegression


def colic_sklearn():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')

    training_set = []
    traing_labels = []
    test_set = []
    test_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(len(curr_line) - 1):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        traing_labels.append(float(curr_line[-1]))

    for line in fr_test.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(len(curr_line) - 1):
            line_arr.append(float(curr_line[i]))
        test_set.append(line_arr)
        test_labels.append(float(curr_line[-1]))
    classifier = LogisticRegression(solver='liblinear', max_iter=50).fit(training_set, traing_labels)
    test_accurcy = classifier.score(test_set, test_labels) * 100
    print('正确率:%f%%' % test_accurcy)


if __name__ == '__main__':
    colic_sklearn()
