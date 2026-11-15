from exp import *
# 1) ceiling: how much of MAE is caused by DNFs (unpredictable)?
h = df[df.season>=2024]
fin = h[h.dnf==0]
print("share of drivers DNF in holdout:", round(h.dnf.mean(),3))
def mae_split(score_fn, label):
    tot, nd = [], []
    for rid in df[df.season>=2024].race_idx.unique():
        g = df[df.race_idx==rid]; tr = df[df.race_idx<rid]
        s = score_fn(tr,g); r = pd.Series(s,index=g.index).rank(method="first")
        e = (r-g.finish_position).abs()
        tot.append(e.mean()); nd.append(e[g.dnf==0].mean())
    print(f"{label:20s} MAE all {np.mean(tot):.3f} | MAE finishers only {np.mean(nd):.3f}")
mae_split(grid,"grid"); mae_split(current,"current")

# 2) direct LightGBM ranking (lambdarank) on all features
def mk_rank(feats, n=150):
    def f(tr,g):
        tr = tr.sort_values("race_idx")
        grp = tr.groupby("race_idx",sort=True).size().values
        rel = (tr.groupby("race_idx").finish_position.transform("max")+1-tr.finish_position).clip(lower=0)
        m = lgb.LGBMRanker(objective="lambdarank",n_estimators=n,learning_rate=0.03,num_leaves=6,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=15,random_state=1,verbose=-1,label_gain=list(range(0,40)))
        m.fit(tr[feats].astype(float), rel.astype(int), group=grp)
        return -m.predict(g[feats].astype(float))
    return f
core = ["grid_position","quali_gap_pct","quali_vs_teammate","grid_penalty","pit_lane_start","team_elo","driver_elo","driver_vs_car_elo","team_qelo","drv_dnf_l10","team_dnf_l10","team_pts_l5","circuit_overtaking","fp_longrun_gap_pct"]
run(mk_rank(core),"lambdarank core")
# 3) lgb alone on core, as delta, with shrink
def mk_lgb(feats, shrink):
    def f(tr,g):
        m = lgb.LGBMRegressor(n_estimators=150,learning_rate=0.03,num_leaves=8,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=15,random_state=1,verbose=-1)
        m.fit(tr[feats].astype(float), tr.finish_position-tr.grid_position)
        return g.grid_position.values + shrink*m.predict(g[feats].astype(float))
    return f
for sh in (0.5,1.0): run(mk_lgb(core,sh),f"lgb core shrink {sh}")
# 4) train only on recent seasons (regulation drift)
def mk_recent(feats,a,years):
    def f(tr,g):
        tr2 = tr[tr.season>=g.season.iloc[0]-years]
        m = ridge(a).fit(tr2[feats].astype(float), tr2.finish_position-tr2.grid_position)
        return g.grid_position.values + m.predict(g[feats].astype(float))
    return f
for y in (3,5): run(mk_recent(core,250,y),f"ridge core, last {y} seasons")
