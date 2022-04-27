h=0.04
#Calculate C(t)
python3 ./c_kawasaki.py -H $h -W 0.5
python3 ./c_kawasaki.py -H $h -W 0.7
python3 ./c_kawasaki.py -H $h -W 0.9
python3 ./c_kawasaki.py -H $h -W 1.0
python3 ./c_kawasaki.py -H $h -W 2.0

#plot
python ./main_plot_beta_dependence_loglog.py -H $h -W 0.5
python ./main_plot_beta_dependence_loglog.py -H $h -W 0.7
python ./main_plot_beta_dependence_loglog.py -H $h -W 0.9
python ./main_plot_beta_dependence_loglog.py -H $h -W 1.0
python ./main_plot_beta_dependence_loglog.py -H $h -W 2.0
