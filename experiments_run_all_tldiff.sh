# TL-difficult subsets (6): 0420, 0183, 0184, 0027, 0264, and 0401

# Process report0420
nohup bash benchmark_sc.sh report0420 > benchmark_sc_0420.out
nohup bash benchmark_tl.sh report0420 > benchmark_tl_0420.out
nohup bash benchmark_tstl.sh report0420 > benchmark_tstl_0420.out
python plot_subset.py report0420

# Process report0183
nohup bash benchmark_sc.sh report0183 > benchmark_sc_0183.out
nohup bash benchmark_tl.sh report0183 > benchmark_tl_0183.out
nohup bash benchmark_tstl.sh report0183 > benchmark_tstl_0183.out
python plot_subset.py report0183

# Process report0184
nohup bash benchmark_sc.sh report0184 > benchmark_sc_0184.out
nohup bash benchmark_tl.sh report0184 > benchmark_tl_0184.out
nohup bash benchmark_tstl.sh report0184 > benchmark_tstl_0184.out 
python plot_subset.py report0184

# Process report0027
nohup bash benchmark_sc.sh report0027 > benchmark_sc_0027.out
nohup bash benchmark_tl.sh report0027 > benchmark_tl_0027.out
nohup bash benchmark_tstl.sh report0027 > benchmark_tstl_0027.out
python plot_subset.py report0027

# Process report0264
nohup bash benchmark_sc.sh report0264 > benchmark_sc_0264.out  
nohup bash benchmark_tl.sh report0264 > benchmark_tl_0264.out 
nohup bash benchmark_tstl.sh report0264 > benchmark_tstl_0264.out
python plot_subset.py report0264

# Process report0401
nohup bash benchmark_sc.sh report0401 > benchmark_sc_0401.out
nohup bash benchmark_tl.sh report0401 > benchmark_tl_0401.out
nohup bash benchmark_tstl.sh report0401 > benchmark_tstl_0401.out 
python plot_subset.py report0401