import pickle as pk

with open('/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/sample_0/allmaps_sim_B12_nside512.pkl', 'rb') as f:
    data = pk.load(f)

print(data.keys())
