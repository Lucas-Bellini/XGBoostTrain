import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


df_cardio = pd.read_csv("cardio_train_new.csv", sep=";", index_col=0)

Y = df_cardio["cardio"]
X = df_cardio.loc[:, df_cardio.columns != 'cardio']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

tree_clf = DecisionTreeClassifier()
xbg_clf = XGBClassifier(learning_rate = 0.07, subsample = 0.6, colsample_bytree = 1,   n_estimators = 92, random_state = 0, min_child_weight = 2, max_depth = 4, objective = "binary:logistic")
rdf_clf = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4)

voting_clf = VotingClassifier(
    estimators = [("lr", xbg_clf),("tree", tree_clf), ("svm", rdf_clf)], voting="hard" 
)

voting_clf.fit(x_train, y_train)

for clf in (xbg_clf, tree_clf, rdf_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


result_cv = cross_val_score(voting_clf, x_train, y_train)
print(f"\nThis is the result of Cross Validation: {result_cv}")
acuracy = result_cv.mean()
print(f"\nWith Cross Validation we had an accuracy of: {acuracy}")


predictions = voting_clf.predict(x_test)


acertos = (predictions == y_test).sum()
total = len(y_test)
print(f"\nThe total number of hits was: {acertos}")
print(f"\nThe total number is: {total}")

acc = accuracy_score(y_test, predictions)

xpredictions = xbg_clf.predict(x_test)
xacc = accuracy_score(y_test, xpredictions)

print(f"\nTherefore, the accuracy of the test model (with accuracy_score as same as VotingClassifier) was: {acc} and the best result (with XGBoostClassifier) was: {xacc}")



