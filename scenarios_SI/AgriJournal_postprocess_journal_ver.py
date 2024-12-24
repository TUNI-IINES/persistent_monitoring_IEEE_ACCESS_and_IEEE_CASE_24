import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import os

#-------------------------------------------------------------------------------------------------------------------------------
# SINGLE INTEGRATOR SCENARIOS
#-------------------------------------------------------------------------------------------------------------------------------
# from scenarios_SI.AgriJournal_scenario import SceneSetup, SimSetup, FeedbackInformation, Controller, ControlOutput, SimSetup, SimulationCanvas

# This file is a modified version of AgriJournal_batch_random_pos.py
# The sole intention for this script is to make beautiful figure
# and additional comparison of the generated batch simulation
# Any basic post-processing part for batch simulation 
# should already be within AgriJournal_batch_random_pos.py

# NOTE: it would be best and easier when some parts of these are integrated in the AgriJournal_batch_random_pos.py
# however, the sensing range sizes are defined manually, this makes it difficult (at the moment).

class BatchRun():
    #----------------------------------------------------------
    # main_folder = 'animation_result/061423_T-ASE/'
    main_folder = 'animation_result/Fikri_research/'

    def __init__(self, argv = []):
        self._argv = argv
        
        # # Process argument
        # val = argv[1]
        # if val == 'gen': self.generate_random_samples()
        # if val == 'post': 
        #     k_val = int(argv[2])
        #     self.post_process_data(k_val)
        # if val == 'post_all': self.post_process_data_all()
        # elif (val.isdigit()):
        #     int_val = int(val)
        #     if int_val < self.total_batch:
        #         # run batch from the input starting index
        #         self.execute(int_val)      

        # MANUAL TESTING FOR NOW
        # from scenarios_SI.AgriJournal_batch_random_pos import BatchRun as br
        # default_path = self.main_folder + 'k_means_uniform_dist_homo'
        # total_iteration = 50
        # n_cluster = 3
        # sr_area = [18., 15., 10., 9., 14., 13.]
        # br.process_batch_K_task_pkl(default_path, total_iteration, n_cluster, sr_area)

        # self.label = [
        #     'uniform_dist_homo',
        #     'clustered_dist_homo',
        #     'uniform_dist_hetero',
        #     'clustered_dist_hetero'
        # ]
        self.label = [
            'NLP1',
            'NLP2',
            'QP1',
            'QP2'
        ]
        scenario_path = [(self.main_folder + 'k_means_' + self.label[i] + '/') for i in range(4)]


        sr_area_hetero = [18., 15., 10., 9., 14., 13.]
        sr_area_homoge = [14., 14., 14., 14., 14., 14.]
        scenario_sr = [ sr_area_homoge, sr_area_homoge, sr_area_hetero, sr_area_hetero]

        for i in range(4):
            # self.post_process_task_scenario(scenario_path[i], scenario_sr[i], i)
            self.post_process_sim_scenario(scenario_path[i], scenario_sr[i], i)


    def post_process_task_scenario(self, scenario_path, scenario_sr, scenario_num):
        from scenarios_SI.AgriJournal_batch_random_pos import BatchRun as br
        
        total_cluster = 6
        robot_num = 6
        total_iteration = 50

        default_path = scenario_path
        sr_area = scenario_sr

        # Initialize matrix to store the data
        # row vs column is number of roi vs each ROI (max is robot_num)
        array_eval_allocate_mean = np.zeros((total_cluster, robot_num))
        array_eval_allocate_std = np.zeros((total_cluster, robot_num))

        array_eval_distance_mean = np.zeros((total_cluster, robot_num))
        array_eval_distance_std = np.zeros((total_cluster, robot_num))
        array_eval_deviate_mean = np.zeros((total_cluster, robot_num))
        array_eval_deviate_std = np.zeros((total_cluster, robot_num))
        
        array_roi_sizes = np.zeros((total_cluster, robot_num))

        for i in range(total_cluster):
            n_cluster = i + 1
            saved_task_area, eval_allocate_mean, eval_allocate_std, \
                eval_distance_mean, eval_distance_std, \
                eval_deviate_mean, eval_deviate_std \
                = br.process_batch_K_task_pkl(default_path, total_iteration, n_cluster, sr_area)

            array_roi_sizes[i,:n_cluster] = saved_task_area
            array_eval_allocate_mean[i,:n_cluster] = eval_allocate_mean
            array_eval_allocate_std[i,:n_cluster] = eval_allocate_std

            array_eval_distance_mean[i,:] = eval_distance_mean
            array_eval_distance_std[i,:] = eval_distance_std
            array_eval_deviate_mean[i,:] = eval_deviate_mean
            array_eval_deviate_std[i,:] = eval_deviate_std

        total_roi_sizes = np.sum(array_roi_sizes, axis=1)
        # TODO: save this info as txt file
        table = np.block([array_roi_sizes, total_roi_sizes.reshape((total_cluster,1))])
        print(np.array2string(table, separator='& '))


        # PLOT Processed data: eval allocation
        # ---------------------------------------
        FS = 14 # font size

        fig, ax = plt.subplots(figsize=(6.4, 3))
        plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True

        X_axis = np.arange(robot_num)
        width, margin = 0.1, 0.11
        spread = np.array( list(range(robot_num)) ) - ((robot_num-1)/2)

        for i in range(robot_num):
            plt.bar(X_axis + spread[i]*margin, array_eval_allocate_mean[:,i], width, \
                    yerr=array_eval_allocate_std[:,i], capsize=2, label = 'j='+str(i+1))
  
        plt.xticks(X_axis, ['$\ell=1$','$\ell=2$','$\ell=3$','$\ell=4$','$\ell=5$','$\ell=6$'])
        plt.ylim((0, 1.1))
        # plt.xlabel("$\ell$")
        plt.ylabel(r'$\varphi^j$')
        # plt.title("Flight Distance Performance")
        # plt.legend()
        save_plot, show_plot = True, False
        if save_plot:
            figname = default_path + 'eval_allocate_' + self.label[scenario_num] + '.pdf'
            plt.savefig(figname, bbox_inches="tight", dpi=300)
            print( 'export figure: ' + figname, flush=True)
        if show_plot: plt.show()


        # PLOT Processed data: eval Distance
        # ---------------------------------------
        FS = 14 # font size

        fig = plt.figure(figsize=(6.4, 3)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        # fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2,1], figsize=(6.4, 3))
        plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True

        X_axis = np.arange(robot_num)
        width, margin = 0.1, 0.11
        spread = np.array( list(range(robot_num)) ) - ((robot_num-1)/2)

        plt.sca(ax1) # DRAW THE FIRST SUBPLOT
        for i in range(robot_num):
            plt.bar(X_axis + spread[i]*margin, array_eval_distance_mean[:,i], width, \
                    yerr=array_eval_distance_std[:,i], capsize=2, label = 'j='+str(i+1))
  
        plt.xticks(X_axis, ['','','','','',''])
        # plt.ylim((0, 1.1))
        # plt.ylabel(r'$||p_i - c^j||$ for $i \in \mathcal{I}^j$')
        plt.ylabel(r'$||p_i - c^j||$')

        plt.sca(ax2) # DRAW THE SECOND SUBPLOT
        for i in range(robot_num):
            plt.bar(X_axis + spread[i]*margin, array_eval_deviate_mean[:,i], width, \
                    yerr=array_eval_deviate_std[:,i], capsize=2, label = 'j='+str(i+1))
  
        plt.xticks(X_axis, ['$\ell=1$','$\ell=2$','$\ell=3$','$\ell=4$','$\ell=5$','$\ell=6$'])
        # plt.ylim((0, 1.1))
        plt.ylabel('deviation \n from min')
        # plt.legend()

        save_plot, show_plot = True, False
        if save_plot:
            figname = default_path + 'eval_distance_' + self.label[scenario_num] + '.pdf'
            plt.savefig(figname, bbox_inches="tight", dpi=300)
            print( 'export figure: ' + figname, flush=True)
        if show_plot: plt.show()



    def post_process_sim_scenario(self, scenario_path, scenario_sr, scenario_num):
        from scenarios_SI.AgriJournal_batch_random_pos import BatchRun as br

        total_cluster = 6
        robot_num = 6
        total_iteration = 50

        default_path = scenario_path
        sr_area = scenario_sr

        # Initialize matrix to store data for coverage
        dict_cluster_n = {(n+1):{} for n in range(total_cluster)}        

        # Initialize matrix to store data for flight distance
        # row vs column is number of roi vs each drone
        array_flight_distance_mean = np.zeros((total_cluster, robot_num))
        array_flight_distance_std = np.zeros((total_cluster, robot_num))

        for i in range(total_cluster):
            n_cluster = i + 1
            # saved_task_area, eval_allocate_mean, eval_allocate_std \
            dict_cluster_n[n_cluster]['time_data'], \
                dict_cluster_n[n_cluster]['holes_cov_ratio_mean'], \
                dict_cluster_n[n_cluster]['holes_cov_ratio_std'], \
                array_flight_distance_mean[i], \
                array_flight_distance_std[i] \
                = br.process_batch_K_sim_pkl(default_path, total_iteration, n_cluster)

        # PLOT Processed data: coverage ratio
        # ---------------------------------------
        FS = 14 # font size

        fig, ax = plt.subplots(figsize=(6.4, 4))
        plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True
        colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # cluster_check = [1, 3, 6] #
        cluster_check = [3] #
        pl_mean = {}
        pl_std = {}
        for n_cluster in cluster_check:
            idx = n_cluster - 1
            time_data = dict_cluster_n[n_cluster]['time_data']
            holes_cov_ratio_mean = dict_cluster_n[n_cluster]['holes_cov_ratio_mean']
            holes_cov_ratio_std = dict_cluster_n[n_cluster]['holes_cov_ratio_std']

            pl_mean[n_cluster], = ax.plot(time_data, holes_cov_ratio_mean, color = colorList[idx], label='all')
            plt.fill_between(time_data, holes_cov_ratio_mean - holes_cov_ratio_std, holes_cov_ratio_mean + holes_cov_ratio_std, 
                             color = colorList[idx], alpha=0.2)
            pl_std[n_cluster], = ax.fill(np.NaN, np.NaN, color = colorList[idx], alpha=0.2) # dummy for legend
        
        ax.set(xlabel="t [s]", ylabel="$\zeta$")
        ax.grid()
        plt.ylim(-0.05, 1.05)
        plt.legend([(pl_mean[i], pl_std[i]) for i in cluster_check], [r'$\ell={}$'.format(i) for i in cluster_check], loc='best')
        save_plot, show_plot = True, False
        if save_plot:
            figname = default_path + 'eval_coverage_' + self.label[scenario_num] + '.pdf'
            plt.savefig(figname, bbox_inches="tight", dpi=300)
            print( 'export figure: ' + figname, flush=True)
        if show_plot: plt.show()

        
        # PLOT Processed data: flight time barchart
        # ---------------------------------------
        FS = 14 # font size

        fig, ax = plt.subplots(figsize=(6.4, 3))
        plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True

        X_axis = np.arange(robot_num)
        width, margin = 0.1, 0.11
        spread = np.array( list(range(robot_num)) ) - ((robot_num-1)/2)

        for i in range(robot_num):
            plt.bar(X_axis + spread[i]*margin, array_flight_distance_mean[:,i], width, \
                    yerr=array_flight_distance_std[:,i], capsize=2, label = 'Quadrotor '+str(i+1))
  
        # plt.xticks(X_axis, ['1','2','3','4','5','6'])
        plt.xticks(X_axis, ['$\ell=1$','$\ell=2$','$\ell=3$','$\ell=4$','$\ell=5$','$\ell=6$'])
        plt.yticks([0, 30, 60, 90, 120])
        plt.ylim((0, 140))
        # plt.xlabel("Number of ROIs")
        plt.ylabel("Total Flight Distances [m]")
        # plt.title("Flight Distance Performance")
        # plt.legend()
        save_plot, show_plot = True, False
        if save_plot:
            figname = default_path + 'eval_dist_' + self.label[scenario_num] + '.pdf'
            # figname = self.def_folder + 'monte_carlo_flight_distances_barchart.pdf'
            plt.savefig(figname, bbox_inches="tight", dpi=300)
            print( 'export figure: ' + figname, flush=True)
        if show_plot: plt.show()

