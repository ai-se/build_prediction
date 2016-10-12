import pandas as pd
import numpy as np
from sklearn import preprocessing

def build_pred(df):
    df = df.drop(['gh_project_name'],axis=1)
    df = df.drop(['gh_is_pr'],axis=1)
    
    y = df['tr_status']
    X = df.drop(['tr_status'],axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    clf = clf.fit(X, y)
    clf_prediction = clf.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print '\nClasification report:\n', classification_report(y_test, clf_prediction)
#     print '\nConfussion matrix:\n', confusion_matrix(y_test, clf_prediction)


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

#data preprocessing
le_gh_is_pr = preprocessing.LabelEncoder()
df.gh_is_pr = le_gh_is_pr.fit_transform(df.gh_is_pr)

le_gh_by_core_team_member = preprocessing.LabelEncoder()
df.gh_by_core_team_member = le_gh_by_core_team_member.fit_transform(df.gh_by_core_team_member)

le_tr_status = preprocessing.LabelEncoder()
df.tr_status = le_tr_status.fit_transform(df.tr_status)

# for one project
proj = pd.value_counts(df.gh_project_name.values).axes[0][0]
count = pd.value_counts(df.gh_project_name.values).values[0]
temp = df[df.gh_project_name == proj]
temp = temp[temp.gh_is_pr == 1]
temp1 = temp[temp.tr_status == 2]
temp2 = temp[temp.tr_status == 3]
frames = [temp1, temp2]
temp = pd.concat(frames)
print "Project: {};   Count: {}".format(proj,count)
build_pred(temp)

# for all projects
# for proj,count in zip(pd.value_counts(df.gh_project_name.values).axes[0],pd.value_counts(df.gh_project_name.values).values):
#     if count > 0:
#         temp = df[df.gh_project_name == proj]
#         print "Project: {};   Count: {}".format(proj,count)
#         build_pred(temp)
