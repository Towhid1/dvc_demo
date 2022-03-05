from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/kidney_disease.csv')
print(data.head())

# print(data.classification.value_counts())
data.classification.replace("ckd\t", "ckd", inplace=True)
# print(data.classification.value_counts())
# print("==="*20)
# print(data.dm.value_counts())
data.dm.replace(["\tno", "\tyes", " yes"], ["no", "yes", "yes"], inplace=True)
# print(data.dm.value_counts())
# print("==="*20)
# print(data.cad.value_counts())
data.cad.replace(["\tno"], ["no"], inplace=True)
# print(data.cad.value_counts())

data.rc.replace("\t?", data.rc.mode()[0], inplace=True)
data.rc = data.rc.apply(lambda x: float(x))

data.wc.replace("\t?", data.wc.mode()[0], inplace=True)
data.wc = data.wc.apply(lambda x: float(x))


data.pcv.replace(["\t?", "\t43"], data.pcv.mode()[0], inplace=True)
data.pcv = data.pcv.apply(lambda x: float(x))

data.classification.replace(["ckd", "notckd"], [1, 0], inplace=True)

filna = data[['id', 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
              'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
              'appet', 'pe', 'ane', 'classification']]

# dealing with missing values
for i in filna.columns:
    if data[i].isna().sum() > 0:
        if data[i].dtype == 'float64':
            data[i].fillna(data[i].median(), inplace=True)
        else:
            data[i].fillna(data[i].mode()[0], inplace=True)
# print(data.head())


def outlinefree(dataCol):
    sorted(dataCol)
    # print(dataCol)
    Q1, Q3 = np.percentile(dataCol, [25, 75])
    IQR = Q3 - Q1
    LowerRange = Q1 - (1.5 * IQR)
    UpperRange = Q3 + (1.5 * IQR)
    return LowerRange, UpperRange


# outlier handle
Lowage, Upage = outlinefree(data.age)
Lowbp, Upbp = outlinefree(data.bp)
Lowsg, Upsg = outlinefree(data.sg)
Lowal, Upal = outlinefree(data.al)
Lowsu, Upsu = outlinefree(data.su)
Lowbgr, Upbgr = outlinefree(data.bgr)
Lowbu, Upbu = outlinefree(data.bu)
Lowsc, Upsc = outlinefree(data.sc)
Lowsod, Upsod = outlinefree(data.sod)
Lowpot, Uppot = outlinefree(data.pot)
Lowhemo, Uphemo = outlinefree(data.hemo)

data.age.replace(list(data[(data.age < Lowage)].age), Lowage, inplace=True)
data.age.replace(list(data[(data.age > Upage)].age), Upage, inplace=True)

data.bp.replace(list(data[(data.bp < Lowbp)].bp), Lowbp, inplace=True)
data.bp.replace(list(data[(data.bp > Upbp)].bp), Upbp, inplace=True)

data.sg.replace(list(data[(data.sg < Lowsg)].sg), Lowsg, inplace=True)
data.sg.replace(list(data[(data.sg > Upsg)].sg), Upsg, inplace=True)

data.al.replace(list(data[(data.al < Lowal)].al), Lowal, inplace=True)
data.al.replace(list(data[(data.al > Upal)].al), Upal, inplace=True)

data.su.replace(list(data[(data.su < Lowsu)].su), Lowsu, inplace=True)
data.su.replace(list(data[(data.su > Upsu)].su), Upsu, inplace=True)

data.bgr.replace(list(data[(data.bgr < Lowbgr)].bgr), Lowbgr, inplace=True)
data.bgr.replace(list(data[(data.bgr > Upbgr)].bgr), Upbgr, inplace=True)

data.bu.replace(list(data[(data.bu < Lowbu)].bu), Lowbu, inplace=True)
data.bu.replace(list(data[(data.bu > Upbu)].bu), Upbu, inplace=True)

data.sc.replace(list(data[(data.sc < Lowsc)].sc), Lowbu, inplace=True)
data.sc.replace(list(data[(data.sc > Upsc)].sc), Upbu, inplace=True)

data.sod.replace(list(data[(data.sod < Lowsod)].sod), Lowsod, inplace=True)
data.sod.replace(list(data[(data.sod > Upsod)].sod), Upsod, inplace=True)

data.pot.replace(list(data[(data.pot < Lowpot)].pot), Lowpot, inplace=True)
data.pot.replace(list(data[(data.pot > Uppot)].pot), Uppot, inplace=True)

data.hemo.replace(
    list(data[(data.hemo < Lowhemo)].hemo), Lowhemo, inplace=True)
data.hemo.replace(list(data[(data.hemo > Uphemo)].hemo), Uphemo, inplace=True)

# print(data.head())

finaldata = data.loc[:, ['classification', 'age', 'bp', 'sg', 'al', 'rbc', 'pc', 'pcc',
                         'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc',
                         'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]

final_dataset = pd.get_dummies(finaldata)
# print(final_dataset.head())

features = final_dataset.iloc[:, 1:].values
label = final_dataset.iloc[:, 0].values

# ------------LogisticRegression----------------
X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size=0.25, random_state=112)

classimodel = LogisticRegression()
classimodel.fit(X_train, y_train)
trainscore = classimodel.score(X_train, y_train)
testscore = classimodel.score(X_test, y_test)

y_pred = classimodel.predict(X_test)
print(classification_report(y_test, y_pred))

# save the model to disk
filename = 'model/LogisticRegression.sav'
joblib.dump(classimodel, filename)
