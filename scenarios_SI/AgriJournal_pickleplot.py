import pickle
import numpy as np
import matplotlib.pyplot as plt

from scenarios_SI.AgriJournal_scenario import SceneSetup, SimSetup #, ExpSetup

def preamble_setting():  # when manual plotting is needed
    SimSetup.sim_fdata_log = './animation_result/061423_T-ASE/k_means_clustered_dist_hetero/k3/d_0/sim_data.pkl'
    SceneSetup.n_cluster = 3
    SceneSetup.robot_num = 6

def extract_single_sim_data(fname, n_cluster, robot_num):
    # ---------------------------------------------------
    with open(fname, 'rb') as f: visData = pickle.load(f)
    __stored_data = visData['stored_data']
    __end_idx = visData['last_idx']

    print('The file ' + fname + ' contains the following logs for ' + '{:.2f}'.format(__stored_data['time'][__end_idx]) + ' s:') 
    print(__stored_data.keys())
    print('with total ' + str(__end_idx) + ' data points')

    # TIME DATA
    # ---------------------------------------------------
    time_data = __stored_data['time'][:__end_idx]

    # COVERAGE METRIC - ENTIRE AREAS
    # ---------------------------------------------------
    dict_total_cov_ratio_each_cluster = {}
    total_cov_current_all = np.zeros( __end_idx )
    total_cov_default_all = np.zeros( __end_idx )
    for i in range(n_cluster):
        dict_total_cov_ratio_each_cluster[i] = __stored_data['cov_total_ratio_' + str(i)][:__end_idx]
        total_cov_current_all += __stored_data['cov_total_current_' + str(i)][:__end_idx]
        total_cov_default_all += __stored_data['cov_total_default_' + str(i)][:__end_idx]
    # Agregate for the all clusters
    total_cov_ratio_all_cluster = total_cov_current_all/total_cov_default_all

    # COVERAGE METRIC - ONLY HOLES IN ENTIRE AREAS
    # ---------------------------------------------------
    dict_holes_cov_ratio_each_cluster = {}
    holes_cov_current_all = np.zeros( __end_idx )
    holes_cov_default_all = np.zeros( __end_idx )
    for i in range(n_cluster):
        dict_holes_cov_ratio_each_cluster[i] = __stored_data['cov_hole_ratio_' + str(i)][:__end_idx]
        holes_cov_current_all += __stored_data['cov_hole_current_' + str(i)][:__end_idx]
        holes_cov_default_all += __stored_data['cov_hole_default_' + str(i)][:__end_idx]
    # Agregate for the all clusters
    holes_cov_ratio_all_cluster = holes_cov_current_all/holes_cov_default_all

    # FLIGHT DISTANCES
    # ---------------------------------------------------
    flight_dist = {i:np.zeros(__end_idx-1) for i in range(robot_num)}
    for i in range(robot_num):
        # Compute each drone flight distance per iteration
        pos_x = __stored_data['pos_x_' + str(i)][:__end_idx]
        pos_y = __stored_data['pos_y_' + str(i)][:__end_idx]
        dist_per_iter = np.array([ np.sqrt((pos_x[j+1]-pos_x[j])**2 + (pos_y[j+1]-pos_y[j])**2) for j in range(__end_idx-1) ])
        # Compute the accumulation of distance over time
        # sum_distance[i] = np.sum(dist_per_iter)
        flight_dist[i][0] = dist_per_iter[0]
        for idx in range(__end_idx-2): flight_dist[i][idx+1] = flight_dist[i][idx] + dist_per_iter[idx+1]
    # Extract final flight distances
    final_distance = [flight_dist[i][-1] for i in range(robot_num)]

    # Compute furthest from initial position
    furthest_distance = [0.]*robot_num
    for i in range(robot_num):
        pos_x = __stored_data['pos_x_' + str(i)][:__end_idx]
        pos_y = __stored_data['pos_y_' + str(i)][:__end_idx]
        dist_to_init = np.array([ np.sqrt((pos_x[j]-pos_x[0])**2 + (pos_y[j]-pos_y[0])**2) for j in range(__end_idx) ])
        furthest_distance[i] = dist_to_init.max()

    # Return the computed information
    return time_data, \
        dict_total_cov_ratio_each_cluster, total_cov_ratio_all_cluster, \
        dict_holes_cov_ratio_each_cluster, holes_cov_ratio_all_cluster, \
        flight_dist, final_distance, furthest_distance


def scenario_pkl_plot():
    # ---------------------------------------------------
    # Extract important data from pickle
    time_data, \
        dict_total_cov_ratio_each_cluster, total_cov_ratio_all_cluster, \
        dict_holes_cov_ratio_each_cluster, holes_cov_ratio_all_cluster, \
        flight_dist, final_distance, furthest_distance \
        = extract_single_sim_data(SimSetup.sim_fdata_log, SceneSetup.n_cluster, SceneSetup.robot_num)


    # ---------------------------------------------------
    figure_short = (6.4, 3.4)
    figure_size = (6.4, 4.8)
    FS = 14 # font size
    LW = 1.5 # line width
    leg_size = 8
    colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # PLOT THE TOTAL COVERAGE METRIC
    # ---------------------------------------------------
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': FS})
    plt.rcParams['text.usetex'] = True

    for i in range(SceneSetup.n_cluster):
        # Plot for each region
        # ax.plot(time_data, dict_total_cov_ratio_each_cluster[i], color = colorList[i], label='j = '+str(i+1)+' (i.e., region $\mathcal{Q}^'+str(i+1)+'$)')
        ax.plot(time_data, dict_total_cov_ratio_each_cluster[i], color = colorList[i], label='Region '+str(i+1), linewidth=3)
    # Plot the agregate
    ax.plot(time_data, total_cov_ratio_all_cluster, color = 'k', label='all', linewidth=3)

    # ax.set(xlabel="t [s]", ylabel="$\zeta^j$ for total coverage")
    ax.set(xlabel="t [s]", ylabel="Coverage performance of all points")
    ax.grid()
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='best')
    #plt.show()

    # Save figure in multiple formats
    base_figname = SimSetup.sim_defname + '_coverage_total_ratio'
    
    # Save as pdf
    plt.savefig(base_figname + '.pdf', bbox_inches="tight", dpi=300)
    print( 'export figure: ' + base_figname  + '.pdf', flush=True)

    # Save as png
    plt.savefig(base_figname + '.png', bbox_inches="tight", dpi=300)
    print( 'export figure: ' + base_figname + '.png', flush=True)

    # PLOT THE HOLE COVERAGE METRIC
    # ---------------------------------------------------
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': FS})
    plt.rcParams['text.usetex'] = True

    for i in range(SceneSetup.n_cluster):
        # Plot for each region
        # ax.plot(time_data, dict_holes_cov_ratio_each_cluster[i], color = colorList[i], label='j = '+str(i+1)+' (i.e., region $\mathcal{Q}^'+str(i+1)+'$)')
        ax.plot(time_data, dict_holes_cov_ratio_each_cluster[i], color = colorList[i], label='Region '+str(i+1))
    # Plot the agregate
    ax.plot(time_data, holes_cov_ratio_all_cluster, color = 'k', label='all Regions')

    # ax.set(xlabel="t [s]", ylabel="$\zeta^j$ for HOLE coverage")
    ax.set(xlabel="t [s]", ylabel="Coverage performance on entire areas")
    ax.grid()
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='best')
    #plt.show()
    # figname = SimSetup.sim_defname+'_coverage_hole_ratio.pdf'
    # plt.savefig(figname, bbox_inches="tight", dpi=300)
    # print( 'export figure: ' + figname, flush=True)

    # Save figure in multiple formats
    base_figname = SimSetup.sim_defname + '_coverage_hole_ratio'
    
    # Save as pdf
    plt.savefig(base_figname + '.pdf', bbox_inches="tight", dpi=300)
    print( 'export figure: ' + base_figname  + '.pdf', flush=True)

    # Save as pdf
    plt.savefig(base_figname + '.png', bbox_inches="tight", dpi=300)
    print( 'export figure: ' + base_figname + '.png', flush=True)

    # PLOT THE FLIGHT DISTANCE
    # ---------------------------------------------------
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']    
    fig, ax = plt.subplots(figsize=(6.4, 3))
    plt.rcParams.update({'font.size': FS})
    plt.rcParams['text.usetex'] = True

    for i in range(SceneSetup.robot_num):
        # Plot the data
        ax.plot(time_data[1:], flight_dist[i], color = color_list[i], label='$i={}$'.format(i+1))

    ax.set(xlabel="t [s]", ylabel="flight distances [m]")
    ax.grid()
    # plt.ylim(-0.1, 100)
    plt.legend(loc='best')
    #plt.show()
    figname = SimSetup.sim_defname+'_flight_dist.pdf'
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    print( 'export figure: ' + figname, flush=True)

    # Print additional metric
    print( 'Each quadrotor total distance in m: ' + str(final_distance), flush=True)    
    print( 'Each quadrotor furthest distance to init_pos in m: ' + str(furthest_distance), flush=True)


def exp_video_pkl_plot():
    pass

