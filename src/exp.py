import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import model_v2 as M
from features_v2 import FEATURES
df = pd.read_csv("../data/processed/model_dataset_v2.csv").sort_values(["race_idx","driver_id"]).reset_index(drop=True)
rids = df.loc[df.season>=2022,"race_idx"].unique()

def ridge(a): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=a))
def metrics(g, score):
    r = pd.Series(score, index=g.index).rank(method="first"); a = g.finish_position
    return (r-a).abs().mean(), spearmanr(r,a)[0], int(r[r==1].index[0]==a.idxmin()) if (a==1).any() else 0

def run(make_pred, label, seasons_sets=((2022,2023),(2024,2025,2026))):
    res = {s:[] for s in seasons_sets}
    for rid in rids:
        g = df[df.race_idx==rid]; sea = g.season.iloc[0]
        for ss in seasons_sets:
            if sea in ss:
                tr = df[df.race_idx<rid]
                res[ss].append(metrics(g, make_pred(tr,g)))
    out = {}
    for ss,v in res.items():
        a = np.array(v); out[ss]=(a[:,0].mean(), a[:,1].mean(), a[:,2].mean())
    print(f"{label:42s} tune MAE {out[seasons_sets[0]][0]:.3f} sp {out[seasons_sets[0]][1]:.3f} | hold MAE {out[seasons_sets[1]][0]:.3f} sp {out[seasons_sets[1]][1]:.3f} win {out[seasons_sets[1]][2]:.3f}")
    return out

def grid(tr,g): return g.grid_position.values
def current(tr,g): return M.score(M.fit(tr),g)
def mk_ridge(feats,a,shrink=1.0):
    def f(tr,g):
        m = ridge(a).fit(tr[feats].astype(float), tr.finish_position-tr.grid_position)
        return g.grid_position.values + shrink*m.predict(g[feats].astype(float))
    return f
if __name__=="__main__":
    run(grid,"grid"); run(current,"current")
    core = ["grid_position","quali_gap_pct","quali_vs_teammate","grid_penalty","pit_lane_start","team_elo","driver_elo","driver_vs_car_elo","team_qelo","drv_dnf_l10","team_dnf_l10","team_pts_l5","circuit_overtaking","fp_longrun_gap_pct"]
    for a in (100,250,1000): run(mk_ridge(core,a),f"ridge core a={a}")
    run(mk_ridge(FEATURES,250,0.5),"ridge all a=250 shrink.5")
    run(mk_ridge(FEATURES,250,1.0),"ridge all a=250 shrink1")
    run(mk_ridge(["grid_position"],1,1.0),"ridge grid only")
