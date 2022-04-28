h=0.01
#Calculate C(t)
python3 ./c_kawasaki.py -H $h 
#plot
python ./main_plot_beta_dependence_loglog.py -H $h
python ./main_plot_beta_dependence_log.py -H $h

