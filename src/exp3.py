from exp import *
import model_v2 as M
orig = (list(M.SMALL), list(M.FEATURES), list(M.DNF_FEATURES))
def with_lists(small, full, dnf):
    def f(tr,g):
        M.SMALL[:] = small; M.FEATURES[:] = full; M.DNF_FEATURES[:] = dnf
        try: return M.score(M.fit(tr), g)
        finally:
            M.SMALL[:], M.FEATURES[:], M.DNF_FEATURES[:] = orig
    return f
drop_groups = {
 "weather/wet": ["drv_wet_vs_dry_gain"],
 "tyre/strategy": ["pit_loss_time","pit_loss_scaled","fast_quali_tyre_wear_penalty","high_deg_circuit","drv_tyre_mgmt_skill"],
 "sprint": ["is_sprint_weekend","sprint_finish","sprint_points","sprint_pos_gain","sprint_vs_grid"],
 "elo rank dupes": ["elo_rank","team_elo_rank","combined_elo","combined_elo_rank","grid_rank_pct"],
 "rolling extras": ["drv_fin_l3","drv_fin_l10","drv_pts_l5","drv_quali_l5","team_fin_l5","team_quali_l5","drv_career_races"],
 "circuit hist": ["drv_circuit_avg_fin"],
 "era": ["reg_era","seasons_since_rules_reset"],
 "fp": ["fp_longrun_gap_pct","fp_race_vs_quali_pace","fp_vs_teammate","fp_has_data","fp_laps_count"],
}
O_S, O_F, O_D = orig
run(with_lists(O_S,O_F,O_D),"current (baseline)")
allx=[]
for k,cols in drop_groups.items():
    f=lambda L,c=cols:[x for x in L if x not in c]
    dnf = O_D if k!="fp" else [x for x in O_D if x!="fp_laps_count"]
    run(with_lists(f(O_S),f(O_F),dnf), f"drop {k}")
    if k!="fp": allx+=cols
f=lambda L:[x for x in L if x not in allx]
run(with_lists(f(O_S),f(O_F),O_D), "SLIM: drop all but fp")
allx2 = allx+drop_groups["fp"]
f2=lambda L:[x for x in L if x not in allx2]
run(with_lists(f2(O_S),f2(O_F),[x for x in O_D if x!="fp_laps_count"]), "SLIM2: drop all incl fp")
print("SLIM small:",f(O_S)); print("SLIM full:",f(O_F))
