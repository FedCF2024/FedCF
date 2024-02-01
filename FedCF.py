import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cluster import OPTICS
from collections import Counter
from package_my.class_inheritance import class_inheritance

n_members = 100
n_forests = 2
betai = [32, 63, 125, 250, 500]
rho = [2, 3, 6, 12, 24]
t1 = [4, 6, 12, 24, 48]
n_outputs = 2

class_inheritance()


def dataLoad(id=0, is_server=False):
    if is_server:
        x_te = np.load("./server/data/uci/adult/x_te.npy")
        y_te = np.load("./server/data/uci/adult/y_te.npy")
        return x_te, y_te
    x_tr = np.load("./devices/device" + str(id) + "/data/uci/adult/100/x_tr.npy")
    y_tr = np.load("./devices/device" + str(id) + "/data/uci/adult/100/y_tr.npy")
    x_val = np.load("./devices/device" + str(id) + "/data/uci/adult/100/x_val.npy")
    y_val = np.load("./devices/device" + str(id) + "/data/uci/adult/100/y_val.npy")
    return x_tr, y_tr, x_val, y_val


def getDistribution():
    distributions = []
    for id in range(1, n_members + 1):
        _, y_tr, _, _ = dataLoad(id)
        num_tr = dict(Counter(y_tr))
        num_array = np.zeros(n_outputs, dtype="int")
        for i in range(n_outputs):
            num_array[i] = num_tr[i]
        distributions.append(num_array)
    return np.array(distributions)


def getCluster():
    distributions = getDistribution()
    optics = OPTICS(min_samples=2)
    optics.fit(distributions)
    zetai = optics.labels_
    return zetai


def getTreesNum():
    dataNum, clientID = [], []
    for id in range(1, n_members + 1):
        _, y_tr, _, _ = dataLoad(id)
        dataNum.append(len(y_tr))
        clientID.append(id)
    indices = [i for i, _ in sorted(enumerate(dataNum), key=lambda x: x[1])]
    interval1 = [clientID[indices[i]] for i in range(40)]
    interval2 = [clientID[indices[i]] for i in range(40, 70)]
    interval3 = [clientID[indices[i]] for i in range(70, 85)]
    interval4 = [clientID[indices[i]] for i in range(85, 95)]
    interval5 = [clientID[indices[i]] for i in range(95, 100)]
    Ki, t1i, rhoi = [], [], []
    for id in range(1, n_members + 1):
        if id in interval1:
            Ki.append(betai[0])
            t1i.append(t1[0])
            rhoi.append(rho[0])
        elif id in interval2:
            Ki.append(betai[1])
            t1i.append(t1[1])
            rhoi.append(rho[1])
        elif id in interval3:
            Ki.append(betai[2])
            t1i.append(t1[2])
            rhoi.append(rho[2])
        elif id in interval4:
            Ki.append(betai[3])
            t1i.append(t1[3])
            rhoi.append(rho[3])
        elif id in interval5:
            Ki.append(betai[4])
            t1i.append(t1[4])
            rhoi.append(rho[4])
    return Ki, t1i, rhoi


def train(x_tr, y_tr, Ki, id, seed):
    mods = []
    for i in range(n_forests):
        rf = RandomForestClassifier(n_estimators=Ki[id - 1], n_jobs=-1, random_state=seed)
        erf = ExtraTreesClassifier(n_estimators=Ki[id - 1], bootstrap=True, n_jobs=-1, random_state=seed)
        rf.fit(x_tr, y_tr)
        mods.append(rf)
        erf.fit(x_tr, y_tr)
        mods.append(erf)
    return mods


def aggregate(mods, t1i, rhoi, zetai, round, x_val_aug):
    # The first selection
    treeBestAll, cmBestAll = [], []
    for i in range(n_members):
        _, _, x_val, y_val = dataLoad(i + 1)
        treeBestOne = []
        cmBestOne = []
        for j in range(n_forests * 2):
            trees = mods[i][j].estimators_
            cmScores = []
            if round == 0:
                for tree in trees:
                    acc, mf1 = tree.score(x_val, y_val)
                    cmScore = (acc + mf1) / 2.0
                    cmScores.append(cmScore)
            else:
                for tree in trees:
                    acc, mf1 = tree.score(x_val_aug[i], y_val)
                    cmScore = (acc + mf1) / 2.0
                    cmScores.append(cmScore)
            indices = np.argsort(cmScores)[::-1][:t1i[i]]

            treeBest, cmBest = [], []
            for indice in indices:
                treeBest.append(trees[indice])
                cmBest.append(cmScores[indice])
            treeBestOne.append(treeBest)
            cmBestOne.append(cmBest)
        treeBestAll.append(treeBestOne)
        cmBestAll.append(cmBestOne)

    # The second selection
    rf0, erf0, accs_rf0, accs_erf0 = {}, {}, {}, {}
    rf1, erf1, accs_rf1, accs_erf1 = {}, {}, {}, {}
    t2m = {}
    for i, zi in enumerate(zetai):
        if zi != -1:
            if zi not in accs_rf0:
                rf0[zi], erf0[zi], accs_rf0[zi], accs_erf0[zi] = [], [], [], []
                rf1[zi], erf1[zi], accs_rf1[zi], accs_erf1[zi] = [], [], [], []
                t2m[zi] = 0
            rf0[zi].extend(treeBestAll[i][0])
            erf0[zi].extend(treeBestAll[i][1])
            rf1[zi].extend(treeBestAll[i][2])
            erf1[zi].extend(treeBestAll[i][3])
            accs_rf0[zi].extend(cmBestAll[i][0])
            accs_erf0[zi].extend(cmBestAll[i][1])
            accs_rf1[zi].extend(cmBestAll[i][2])
            accs_erf1[zi].extend(cmBestAll[i][3])
            t2m[zi] += rhoi[i]
    zmax = max(zetai)
    for i, zi in enumerate(zetai):
        if zi == -1:
            zmax += 1
            rf0[zmax], erf0[zmax], accs_rf0[zmax], accs_erf0[zmax] = [], [], [], []
            rf1[zmax], erf1[zmax], accs_rf1[zmax], accs_erf1[zmax] = [], [], [], []
            rf0[zmax].extend(treeBestAll[i][0])
            erf0[zmax].extend(treeBestAll[i][1])
            rf1[zmax].extend(treeBestAll[i][2])
            erf1[zmax].extend(treeBestAll[i][3])
            accs_rf0[zmax].extend(cmBestAll[i][0])
            accs_erf0[zmax].extend(cmBestAll[i][1])
            accs_rf1[zmax].extend(cmBestAll[i][2])
            accs_erf1[zmax].extend(cmBestAll[i][3])
            t2m[zmax] = rhoi[i]
    zmax += 1
    treeBestAll = []
    for i in range(zmax):
        treeBestOne = []
        indices0 = np.argsort(accs_rf0[i])[::-1][:t2m[i]]
        indices1 = np.argsort(accs_erf0[i])[::-1][:t2m[i]]
        indices2 = np.argsort(accs_rf1[i])[::-1][:t2m[i]]
        indices3 = np.argsort(accs_erf1[i])[::-1][:t2m[i]]
        for j in range(n_forests * 2):
            treeBest = []
            if j == 0:
                for indice in indices0:
                    treeBest.append(rf0[i][indice])
            elif j == 1:
                for indice in indices1:
                    treeBest.append(erf0[i][indice])
            elif j == 2:
                for indice in indices2:
                    treeBest.append(rf1[i][indice])
            elif j == 3:
                for indice in indices3:
                    treeBest.append(erf1[i][indice])
            treeBestOne.append(treeBest)
        if i == 0:
            treeBestAll = treeBestOne
        else:
            for j in range(n_forests * 2):
                treeBestAll[j] += treeBestOne[j]
    # Aggregation
    modAgg = []
    for i in range(n_forests * 2):
        if i % 2 == 0:
            mod = RandomForestClassifier(n_estimators=betai[-1], n_jobs=-1)
        else:
            mod = ExtraTreesClassifier(n_estimators=betai[-1], n_jobs=-1)
        mod.n_classes_ = mods[0][0].n_classes_
        mod.n_outputs_ = mods[0][0].n_outputs_
        mod.classes_ = mods[0][0].classes_
        mod.estimators_ = treeBestAll[i]
        modAgg.append(mod)
    return modAgg


def test(X, Y, mod, get_acc=False):
    acc, macro_f1 = [], []
    aug_feature = np.zeros((X.shape[0], n_forests * 2 * n_outputs), dtype="float64")
    for i in range(len(mod)):
        pred_y, class_sector = mod[0].predict(X)
        if get_acc:
            acc.append(accuracy_score(Y, pred_y))
            _, _, mf1, _ = precision_recall_fscore_support(Y, pred_y, average='macro', zero_division=1)
            macro_f1.append(mf1)
        aug_feature[:, i * n_outputs:(i + 1) * n_outputs] = class_sector
    if get_acc:
        ave_acc = np.mean(acc)
        ave_macro_f1 = np.mean(macro_f1)
        return ave_acc, ave_macro_f1, aug_feature
    return aug_feature


if __name__ == '__main__':
    max_acc, max_macro_f1 = 0.0, 0.0
    seed = 1

    zetai = getCluster()
    Ki, t1i, rhoi = getTreesNum()
    for r in range(25):
        mods = []
        if r == 0:
            for id in range(1, n_members + 1):
                x_tr, y_tr, x_val, y_val = dataLoad(id)
                mod = train(x_tr, y_tr, Ki, id, seed)
                mods.append(mod)
        else:
            for id in range(1, n_members + 1):
                x_tr, y_tr, x_val, y_val = dataLoad(id)
                mod = train(x_tr_aug[id - 1], y_tr, Ki, id, seed)
                mods.append(mod)
        if r == 0:
            modAgg = aggregate(mods, t1i, rhoi, zetai, r, None)
        else:
            modAgg = aggregate(mods, t1i, rhoi, zetai, r, x_val_aug)
        if r != 0:
            x_te_aug_old = x_te_aug
            x_tr_aug_old = x_tr_aug
            x_val_aug_old = x_val_aug
        x_te, y_te = dataLoad(is_server=True)
        if r == 0:
            acc, macro_f1, aug_feature_te = test(x_te, y_te, modAgg, get_acc=True)
        else:
            acc, macro_f1, aug_feature_te = test(x_te_aug_old, y_te, modAgg, get_acc=True)
        x_te_aug = np.concatenate((x_te, aug_feature_te), axis=1)
        if acc > max_acc:
            max_acc = acc
        if macro_f1 > max_macro_f1:
            max_macro_f1 = macro_f1
        print("\nacc：%.4f" % acc)
        print("macro_f1：%.4f" % macro_f1)
        x_tr_aug, x_val_aug = [], []
        for id in range(1, n_members + 1):
            x_tr, y_tr, x_val, y_val = dataLoad(id)
            if r == 0:
                aug_feature_tr = test(x_tr, y_tr, modAgg)
                aug_feature_val = test(x_val, y_val, modAgg)
            else:
                aug_feature_tr = test(x_tr_aug_old[id - 1], y_tr, modAgg)
                aug_feature_val = test(x_val_aug_old[id - 1], y_val, modAgg)
            x_tr_aug.append(np.concatenate((x_tr, aug_feature_tr), axis=1))
            x_val_aug.append(np.concatenate((x_val, aug_feature_val), axis=1))
