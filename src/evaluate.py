"""
evaluate.py  —  Reproduce all paper tables
PhishGuard-URL (Molefi, 2026)

Usage:
  python src/evaluate.py                  # full evaluation
  python src/evaluate.py --skip_svm       # skip SVM (faster)
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
SEED = 42

def load_split(path):
    df = pd.read_csv(path)
    y  = df["label"].values
    X  = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values
    return X, y

def metrics(yt, yp, yprob):
    acc=accuracy_score(yt,yp); prec=precision_score(yt,yp,zero_division=0)
    rec=recall_score(yt,yp,zero_division=0); f1=f1_score(yt,yp,zero_division=0)
    auc=roc_auc_score(yt,yprob)
    tn,fp,fn,tp=confusion_matrix(yt,yp).ravel()
    fpr=fp/(fp+tn) if (fp+tn) else 0
    return [f"{100*acc:.2f}%",f"{100*prec:.2f}%",f"{100*rec:.2f}%",
            f"{100*f1:.2f}%",f"{auc:.4f}",f"{100*fpr:.2f}%"]

def ptable(title, rows, headers):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")
    w=[max(len(h),max(len(str(r[i]))for r in rows))for i,h in enumerate(headers)]
    fmt="  ".join(f"{{:<{x}}}"for x in w)
    print(fmt.format(*headers)); print("-"*72)
    for r in rows: print(fmt.format(*[str(v)for v in r]))
    print("="*72)

def evaluate(test_path, train_path, models_dir, skip_svm):
    print("\nLoading data ...")
    X_test, y_test   = load_split(test_path)
    X_train, y_train = load_split(train_path)
    print(f"  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}   Features: {X_test.shape[1]}")

    print("Loading pre-trained models ...")
    scaler = joblib.load(f"{models_dir}/scaler.joblib")
    rf     = joblib.load(f"{models_dir}/rf_model.joblib")
    gb     = joblib.load(f"{models_dir}/gb_model.joblib")
    lr     = joblib.load(f"{models_dir}/lr_model.joblib")
    with open(f"{models_dir}/fusion_weights.json") as f: w=json.load(f)
    w_rf,w_gb,w_lr = w["w_RF"],w["w_GB"],w["w_LR"]
    Xts = scaler.transform(X_test)

    rows=[]
    models_list = [
        ("Naive Bayes",          lambda: (GaussianNB(), X_train, X_test)),
        ("Logistic Regression",  None),
        ("SVM (RBF)",            None),
        ("Decision Tree",        lambda: (DecisionTreeClassifier(max_depth=20,random_state=SEED), X_train, X_test)),
        ("k-NN (k=5)",           None),
        ("Random Forest",        None),
        ("Gradient Boosting",    None),
        ("PhishGuard (Ensemble)",None),
    ]

    for i,(name,_) in enumerate(models_list,1):
        print(f"  [{i}/8] {name} ...", end=" ", flush=True); t0=time.time()

        if name=="Naive Bayes":
            m=GaussianNB(); m.fit(X_train,y_train)
            rows.append((name,*metrics(y_test,m.predict(X_test),m.predict_proba(X_test)[:,1])))

        elif name=="Logistic Regression":
            sc=StandardScaler(); m=LogisticRegression(max_iter=1000,random_state=SEED)
            m.fit(sc.fit_transform(X_train),y_train)
            rows.append((name,*metrics(y_test,m.predict(sc.transform(X_test)),m.predict_proba(sc.transform(X_test))[:,1])))

        elif name=="SVM (RBF)":
            if skip_svm:
                print("SKIPPED (--skip_svm)"); rows.append((name,"—","—","—","—","—","—")); continue
            print("(slow, ~10-15 min on large datasets) ", end=" ", flush=True)
            sc=StandardScaler(); m=SVC(kernel="rbf",C=1.0,probability=True,random_state=SEED)
            m.fit(sc.fit_transform(X_train),y_train)
            rows.append((name,*metrics(y_test,m.predict(sc.transform(X_test)),m.predict_proba(sc.transform(X_test))[:,1])))

        elif name=="Decision Tree":
            m=DecisionTreeClassifier(max_depth=20,random_state=SEED); m.fit(X_train,y_train)
            rows.append((name,*metrics(y_test,m.predict(X_test),m.predict_proba(X_test)[:,1])))

        elif name=="k-NN (k=5)":
            sc=StandardScaler(); m=KNeighborsClassifier(n_neighbors=5)
            m.fit(sc.fit_transform(X_train),y_train)
            rows.append((name,*metrics(y_test,m.predict(sc.transform(X_test)),m.predict_proba(sc.transform(X_test))[:,1])))

        elif name=="Random Forest":
            rows.append((name,*metrics(y_test,rf.predict(X_test),rf.predict_proba(X_test)[:,1])))

        elif name=="Gradient Boosting":
            rows.append((name,*metrics(y_test,gb.predict(X_test),gb.predict_proba(X_test)[:,1])))

        elif name=="PhishGuard (Ensemble)":
            p=w_rf*rf.predict_proba(X_test)[:,1]+w_gb*gb.predict_proba(X_test)[:,1]+w_lr*lr.predict_proba(Xts)[:,1]
            rows.append((name,*metrics(y_test,(p>=0.5).astype(int),p)))

        print(f"done ({time.time()-t0:.1f}s)")

    ptable("Table 3: Overall Detection Performance (Test Set)",rows,
           ["Model","Accuracy","Precision","Recall","F1","AUC-ROC","FPR"])

    # Confusion matrix
    p_ens=w_rf*rf.predict_proba(X_test)[:,1]+w_gb*gb.predict_proba(X_test)[:,1]+w_lr*lr.predict_proba(Xts)[:,1]
    y_ens=(p_ens>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_test,y_ens).ravel()
    print(f"\n{'='*50}\n  Table 4: Confusion Matrix\n{'='*50}")
    print(f"  {'':22} Pred Legit  Pred Phish")
    print(f"  {'True Legitimate':<22} {tn:<11} {fp}")
    print(f"  {'True Phishing':<22} {fn:<11} {tp}")
    print(f"  FPR: {100*fp/(fp+tn):.2f}%   FNR: {100*fn/(fn+tp):.2f}%")
    print("="*50)

    # 10-fold CV
    print("\nRunning 10-fold cross-validation (RF) — 2-3 minutes ...")
    Xa=np.vstack([X_train,X_test]); ya=np.concatenate([y_train,y_test])
    rf_cv=RandomForestClassifier(n_estimators=200,min_samples_split=5,
                                  max_features="sqrt",random_state=SEED,n_jobs=-1)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)
    accs=cross_val_score(rf_cv,Xa,ya,cv=cv,scoring="accuracy",n_jobs=-1)
    print(f"\n{'='*55}\n  Table 6: 10-Fold Cross-Validation (RF)\n{'='*55}")
    for i,a in enumerate(accs,1): print(f"  Fold {i:>2}: {100*a:.2f}%")
    print(f"  Mean: {100*accs.mean():.2f}% ± {100*accs.std():.2f}%  "
          f"Range: {100*accs.min():.2f}%–{100*accs.max():.2f}%")
    print("="*55)

    # Ablation
    print("\nRunning ablation study ...")
    groups={"Entropy (-13)":list(range(64,77)),"Length (-8)":list(range(0,8)),
            "Symbol/Digit (-19)":list(range(45,64)),"Ratio (-16)":list(range(15,31)),
            "Token (-14)":list(range(31,45))}
    gb_f=GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=5,random_state=SEED)
    gb_f.fit(X_train,y_train)
    ba=accuracy_score(y_test,gb_f.predict(X_test))
    bf=f1_score(y_test,gb_f.predict(X_test))
    ab_rows=[("Full model (77 features)",77,f"{100*ba:.2f}%",f"{100*bf:.2f}%","—")]
    all_f=set(range(X_train.shape[1]))
    for gn,ri in groups.items():
        keep=sorted(all_f-set(ri))
        gb_a=GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=5,random_state=SEED)
        gb_a.fit(X_train[:,keep],y_train)
        aa=accuracy_score(y_test,gb_a.predict(X_test[:,keep]))
        af=f1_score(y_test,gb_a.predict(X_test[:,keep]))
        ab_rows.append((f"w/o {gn}",len(keep),f"{100*aa:.2f}%",f"{100*af:.2f}%",f"{100*(aa-ba):+.2f}%"))
    ptable("Table 7: Ablation Study (GB)",ab_rows,["Configuration","Feats","Acc","F1","ΔAcc"])

    print("\n✅  All paper tables reproduced successfully.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--test",     default="data/test.csv")
    p.add_argument("--train",    default="data/train.csv")
    p.add_argument("--models",   default="models/")
    p.add_argument("--skip_svm", action="store_true",
                   help="Skip SVM baseline (saves ~15 min)")
    a=p.parse_args()
    for f in [a.test,a.train]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run dataset_split.py and train.py first.")
            sys.exit(1)
    evaluate(a.test,a.train,a.models,a.skip_svm)
