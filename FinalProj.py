import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn import preprocessing
import Levenshtein
ltable = pd.read_csv("ltable.csv")
rtable = pd.read_csv("rtable.csv")
training = pd.read_csv("train.csv")

def preprocess(row):
    for i in list((np.where(row.brand.isnull().values==True))[0]):
        row.at[i,"brand"] = row.iloc[i].title.split()[0]
preprocess(ltable)
preprocess(rtable)

merged = ltable.merge(rtable, on = "brand")
merged["modelno_x"] = merged["modelno_x"].astype(str).str.lower()
merged["modelno_y"] = merged["modelno_y"].astype(str).str.lower()
merged["category_x"] = merged["category_x"].astype(str)
merged["category_y"] = merged["category_y"].astype(str)

#bag of words model
matches = []
total = []
def jaccard(row, attribute):
    x = set(row[attribute + "_x"].lower().split())
    y = set(row[attribute + "_y"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))
def Ratcliffe(row, attribute):
    x = (row[attribute + "_x"].lower())
    y = (row[attribute + "_y"].lower())
    return max([SequenceMatcher(None, x, y).ratio(),SequenceMatcher(None, y, x).ratio()])
def Lev (row, attribute):
    x = (row[attribute + "_x"].lower())
    y = (row[attribute + "_y"].lower())
    return max([Levenshtein.distance(x, y), Levenshtein.distance(y, x)])
def price(row, attribute):
    x = (row[attribute + "_x"])
    y = (row[attribute + "_y"])
    return (np.abs(x-y) / (x+y))
Scaler = preprocessing.StandardScaler()
def featureEngineering(df):
    atts = ["title", "category", "modelno"]
    for i in atts:
        ratcliffeScore = Scaler.fit_transform(df.apply(Ratcliffe, attribute = i, axis = 1).to_numpy().reshape((-1,1)))
        jaccardScore = Scaler.fit_transform(df.apply(jaccard, attribute = i, axis = 1).to_numpy().reshape((-1,1)))
        LevScore = Scaler.fit_transform(df.apply(Lev, attribute = i, axis = 1).to_numpy().reshape((-1,1)))
        df[("ratcliffeScore" + i)] = ratcliffeScore
        df[("jaccardScore"+i)] = jaccardScore
        df[("LevScore"+i)] = LevScore
    df["PriceDifference"] = Scaler.fit_transform(df.apply(price, attribute = "price", axis = 1).to_numpy().reshape((-1,1)))
    return df
merged = featureEngineering(merged)

trainingindicies = []
traininglabels = []
merged.drop(["title_x", "category_x", "brand","modelno_x", "title_y", "category_y", "modelno_y", "LevScorecategory"], axis = 1, inplace= True)
for i in merged.columns:
    merged[i].fillna(np.mean(merged[i]), inplace= True)
for i in training.iterrows():
    i = i[1]
    xid = i["ltable_id"]
    yid = i["rtable_id"]
    label = i["label"]
    z = (merged.iloc[np.where((merged['id_x'] == xid) & (merged["id_y"] == yid))])
    if z.shape[0] > 0:
        trainingindicies.append(z.index.to_list()[0])
        traininglabels.append(label)
trainingDF = pd.DataFrame(merged.iloc[trainingindicies])
trainingDF["labels"] = traininglabels

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
SVM = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', C = 6, cache_size=9000, probability=True, break_ties= True), max_samples=1.0 / 15, n_estimators=15), n_jobs = -1)
#neural = SVC(kernel="linear", C=25, cache_size= 3000, break_ties= True, probability=True)#class_weight="balanced")
SVM.fit(trainingDF.drop(["labels"], axis=1), traininglabels)
y_pred = SVM.predict(merged)
matching_pairs = merged.loc[y_pred == 1, ["id_x", "id_y"]]
matching_pairs = list(map(tuple, matching_pairs.values))
matching_pairs_in_training = trainingDF.loc[trainingDF["labels"] == 1, ["id_x", "id_y"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)
