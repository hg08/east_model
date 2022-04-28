h=0.01
#Calcuate C(t)
python3 ./c_EJ.py -H $h 

#plot
python3 ./main_plot_beta_dependence_loglog.py -H $h 
