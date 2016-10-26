import pandas as pd
import numpy as np
from sklearn import preprocessing
import time

def build_pred_split(df):
    df = df.drop(['gh_project_name'],axis=1)
    df = df.drop(['gh_is_pr'],axis=1)
    
    y = df['tr_status']
    X = df.drop(['tr_status'],axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    clf = clf.fit(X_train, y_train)
    clf_prediction = clf.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print '\nClasification report:\n', classification_report(y_test, clf_prediction)
    print '\nConfussion matrix:\n', confusion_matrix(y_test, clf_prediction)

def build_pred_cv_old(df):
    df = df.drop(['gh_project_name'],axis=1)
    df = df.drop(['gh_is_pr'],axis=1)
    
    y = df['tr_status']
    X = df.drop(['tr_status'],axis=1)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)

    all_y = []
    all_clf = []
    
    for traincv, testcv in kf.split(df,y=y):
        clf_pred = clf.fit(X.iloc[traincv], y.iloc[traincv]).predict(X.iloc[testcv])
        all_y.append(y.iloc[testcv])
        all_clf.append(clf_pred)
        
    big_y = pd.concat(all_y)
    big_clf = np.concatenate(all_clf)
    from sklearn.metrics import classification_report, confusion_matrix
    print '\nClasification report:\n', classification_report(big_y, big_clf, target_names = ['Erroed', 'Canceled', 'Failed', 'Passed'])
    print '\nConfussion matrix:\n', confusion_matrix(big_y, big_clf)

def build_pred(df, clf):
    df = df.drop(['gh_project_name'],axis=1)
    df = df.drop(['gh_is_pr'],axis=1)
    
    y = df['tr_status']
    X = df.drop(['tr_status'],axis=1)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)

    all_y = []
    all_clf = []
    
    for traincv, testcv in kf.split(df,y=y):
        clf_pred = clf.fit(X.iloc[traincv], y.iloc[traincv]).predict(X.iloc[testcv])
        all_y.append(y.iloc[testcv])
        all_clf.append(clf_pred)
        
    big_y = pd.concat(all_y)
    big_clf = np.concatenate(all_clf)
    # from sklearn.metrics import classification_report, confusion_matrix
    # print 'Clasification report:\n', classification_report(big_y, big_clf)
    from sklearn.metrics import precision_recall_fscore_support
    print  precision_recall_fscore_support(big_y, big_clf, average='weighted')
    # print '\nConfussion matrix:\n', confusion_matrix(big_y, big_clf)

def read_data():
    #reading data
    df = pd.read_csv("/Users/grferrei/Documents/data/travistorrent-5-3-2016.csv",dtype=str)
    df = df.drop(['row','git_commit','git_merged_with', 'gh_lang',
            'git_branch', 'gh_first_commit_created_at','git_commits','gh_description_complexity','tr_build_id','gh_pull_req_num',
            'tr_duration', 'tr_started_at', 'tr_jobs',
            'tr_build_number', 'tr_job_id', 'tr_lan', 'tr_setup_time',
            'tr_analyzer', 'tr_frameworks', 'tr_tests_ok', 'tr_tests_fail',
            'tr_tests_run', 'tr_tests_skipped', 'tr_failed_tests',
            'tr_testduration', 'tr_purebuildduration', 'tr_tests_ran',
            'tr_tests_failed', 'git_num_committers', 'tr_num_jobs',
            'tr_prev_build', 'tr_ci_latency'],axis=1)
    return df

def pre_process(df):
    #data preprocessing
    le_gh_is_pr = preprocessing.LabelEncoder()
    df.gh_is_pr = le_gh_is_pr.fit_transform(df.gh_is_pr)

    le_gh_by_core_team_member = preprocessing.LabelEncoder()
    df.gh_by_core_team_member = le_gh_by_core_team_member.fit_transform(df.gh_by_core_team_member)

    le_tr_status = preprocessing.LabelEncoder()
    df.tr_status = le_tr_status.fit_transform(df.tr_status)

    return df

def single_project_pred(df, clf):
    proj = pd.value_counts(df.gh_project_name.values).axes[0][10]
    count = pd.value_counts(df.gh_project_name.values).values[10]
    temp = df[df.gh_project_name == proj]
    # temp = temp[temp.gh_is_pr == 1]
    # temp = pd.concat([temp[temp.tr_status == 2], temp[temp.tr_status == 3]])
    print "Project: {};   Count: {}".format(proj,count)
    build_pred(temp, clf)

def all_projects_pred(df, clf):
    for proj,count in zip(pd.value_counts(df.gh_project_name.values).axes[0],pd.value_counts(df.gh_project_name.values).values):
        if count > 0:
            temp = df[df.gh_project_name == proj]
            print "Project: {};   Count: {}".format(proj,count)
            build_pred(temp, clf)

def run_all(df, clf):

    build_pred(df, clf)

def classifiers():
    clfs = []

    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier()
    clfs.append(clf)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clfs.append(clf)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clfs.append(clf)

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clfs.append(clf)

    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters = 4)
    clfs.append(clf)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier()
    clfs.append(clf)

    # from sklearn.svm import SVC
    # clf = SVC()
    # clfs.append(clf)

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clfs.append(clf)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clfs.append(clf)

    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()
    clfs.append(clf)

    from sklearn.ensemble import BaggingClassifier
    clf = BaggingClassifier()
    clfs.append(clf)

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    clfs.append(clf)


    return clfs

if __name__ == "__main__":
    # print (time.strftime("%H:%M:%S"))
    df = read_data()
    df = pre_process(df)
    clfs = classifiers()
    for clf in clfs:
        print str(clf)[:str(clf).find('(')]
        t1 = time.time()
        run_all(df, clf)
        # print (time.strftime("%H:%M:%S"))
        print (time.time() - t1)



