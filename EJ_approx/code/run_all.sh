h=0.04
#Calcuate C(t)
#python3 ./c_EJ.py -H $h -W 0.1
python3 ./c_EJ.py -H $h -W 0.3
#python3 ./c_EJ.py -H $h -W 0.5
python3 ./c_EJ.py -H $h -W 0.7
python3 ./c_EJ.py -H $h -W 0.9
python3 ./c_EJ.py -H $h -W 1.0
python3 ./c_EJ.py -H $h -W 2.0

#plot
#python3 ./main_plot_beta_dependence_loglog.py -H $h -W 0.1
python3 ./main_plot_beta_dependence_loglog.py -H $h -W 0.3
#python3 ./main_plot_beta_dependence_loglog.py -H $h -W 0.5
python3 ./main_plot_beta_dependence_loglog.py -H $h -W 0.7
python3 ./main_plot_beta_dependence_loglog.py -H $h -W 0.9
python3 ./main_plot_beta_dependence_loglog.py -H $h -W 1.0
python3 ./main_plot_beta_dependence_loglog.py -H $h -W 2.0
