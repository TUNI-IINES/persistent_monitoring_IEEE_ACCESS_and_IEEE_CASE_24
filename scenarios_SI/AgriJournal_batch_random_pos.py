import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os

#-------------------------------------------------------------------------------------------------------------------------------
# SINGLE INTEGRATOR SCENARIOS
#-------------------------------------------------------------------------------------------------------------------------------
from scenarios_SI.AgriJournal_scenario import SceneSetup, SimSetup, FeedbackInformation, Controller, ControlOutput, SimSetup, SimulationCanvas


#----------------------------------------------------------
# This part is identical with the Simulate in Sim2D_main.py
class Simulate():
    def __init__(self):
        # Initialize components
        self.environment = SimulationCanvas() # Always the first to call, define main setup
        self.controller_block = Controller() 
        # Initialize messages to pass between blocks
        self.feedback_information = FeedbackInformation() 
        self.control_input = ControlOutput() 

    # MAIN LOOP CONTROLLER & VISUALIZATION
    def loop_sequence(self, i = 0):
        # Showing Time Stamp
        if (i > 0) and (i % round(1/SimSetup.Ts) == 0):
            t = round(i*SimSetup.Ts)
            if t < SimSetup.tmax: print('simulating t = {}s.'.format(t))

        # Compute control input and advance simulation
        self.controller_block.compute_control( self.feedback_information, self.control_input )
        self.environment.update_simulation( self.control_input, self.feedback_information )
#----------------------------------------------------------


class BatchRun():
    #----------------------------------------------------------
    # main_folder = 'animation_result/fikriConf_batch_random_pos/'
    main_folder = 'animation_result/fikriJournal_batch_random_pos/'
    # main_folder = 'animation_result/all_clusters/'

    # ALL SCENARIOS SETTINGS
    def_folder = main_folder + 'k_means_uniform_dist/'
    # def_folder = main_folder + 'k_median_uniform_dist_hetero/'
    CLUSTERING_MODE = 0 # clustering broken sensors with K-means
    GENERATE_SENSOR_MODE = 0 # generate a uniform broken sensor

    # def_folder = main_folder + 'k_means_clustered_dist/'
    # CLUSTERING_MODE = 0 # clustering broken sensors with K-means
    # GENERATE_SENSOR_MODE = 1 # generate with a clustered broken sensor

    # def_folder = main_folder + 'fuzzy_cmeans_uniform_dist/'
    # CLUSTERING_MODE = 2 # clustering broken sensors with Fuzzy c-means
    # GENERATE_SENSOR_MODE = 0 # generate with a uniform broken sensor

    # def_folder = main_folder + 'fuzzy_cmeans_clustered_dist/'
    # CLUSTERING_MODE = 2 # clustering broken sensors with Fuzzy c-means
    # GENERATE_SENSOR_MODE = 1 # generate with a clustered broken sensor

    #----------------------------------------------------------

    # Batch Parameter
    total_batch = 50
    iter_per_batch = 50

    # We use the same generated position data for all scenarios
    pkl_random_data = main_folder + 'generated_random_data.pkl'

    # Set the varying parameters here
    currently_tested_n_cluster = 1
    currently_tested_simulation_time = 2


    def __init__(self, argv = []):
        self._argv = argv
        
        # Process argument
        val = argv[1]
        # if val == 'gen': self.generate_random_samples()
        if val == 'gen': self.generate_random_sensing_samples()
        if val == 'post': 
            k_val = int(argv[2])
            self.post_process_data(k_val)
        if val == 'post_all': self.post_process_data_all()
        elif (val.isdigit()):
            int_val = int(val)
            if int_val < self.total_batch:
                # run batch from the input starting index
                self.execute(int_val)      

    # TODO: Finishing random sampling based on sensing range
    def generate_random_sensing_samples(self):
        random_init_size = {}
        random_init = {}
        for i in range(self.total_batch):
            # Define the range for random integers
            min_value = 35  # Minimum value (inclusive)
            max_value = 60  # Maximum value (inclusive)

            # Generate a list of unique random integers
            # create initial sensing range array
            # init_size = random.sample(range(min_value, max_value + 1), 6)
            init_size = random.sample(range(min_value, max_value + 1), 10)
            random_init_size[i] = init_size.copy()
            init_pos = np.array([[11., 12., 0.], [6., 8., 0.], [6., 10., 0.], [14., 7., 0.], [10., 14., 0.], [17., 3., 0.],
                        [1., 1., 0.], [15., 5., 0.], [10., 18., 0.], [7., 14., 0.]])
            # init_pos = np.array([[11., 12., 0.], [6., 8., 0.], [6., 10., 0.], [14., 7., 0.], [10., 14., 0.], [20., 3., 0.]])
            random_init[i] = init_pos.copy()

        # self.set_directory(self.def_folder)
        self.set_directory(self.main_folder)
        # save to pickle
        with open(self.pkl_random_data, 'wb') as f:
            pickle.dump(dict(batch_init=random_init_size, n=self.total_batch), f)
        print('Finished generating ' + str(self.total_batch) + ' random initial sensing range into: ' \
            + self.pkl_random_data, flush=True)
        
    def generate_random_samples(self):
        random_init = {}
        for i in range(self.total_batch):
            # create initial position array
            init_pos = np.random.uniform(low=0, high=20, size=(SceneSetup.robot_num, 2))
            # init_pos = np.hstack((init_pos, np.zeros((6,1))))
            init_pos = np.hstack((init_pos, np.zeros((10,1))))
            random_init[i] = init_pos.copy()

        # self.set_directory(self.def_folder)
        self.set_directory(self.main_folder)
        # save to pickle
        with open(self.pkl_random_data, 'wb') as f:
            pickle.dump(dict(batch_init=random_init, n=self.total_batch), f)
        print('Finished generating ' + str(self.total_batch) + ' random initial positions into: ' \
            + self.pkl_random_data, flush=True)


    def execute(self, init_idx=0):
        # Read the pickle
        with open(self.pkl_random_data, 'rb') as f: Data = pickle.load(f)
        batch_init = Data['batch_init']
        data_num = Data['n']

        # TODO: some manual checking if data_num is the same (or exceed) self.total_batch
        end_idx = min( init_idx+self.iter_per_batch, self.total_batch )
        print('start simulating for random data id: ' + str(init_idx) + '-' + str(end_idx-1))
        print('running for ' + str(self.currently_tested_n_cluster) + ' number of clusters, for ' \
              + str(self.currently_tested_simulation_time) + ' seconds each.\n' )

        for i in range(init_idx, end_idx):
            # ADJUST SCENARIO PARAMETER
            SceneSetup.CLUSTERING_MODE = self.CLUSTERING_MODE
            SceneSetup.GENERATE_SENSOR_MODE = self.GENERATE_SENSOR_MODE

            # ADJUST THE SIMULATION PARAMETER
            SceneSetup.sr_area = batch_init[i].copy()
            # SceneSetup.init_pos = batch_init[i].copy()
            SceneSetup.n_cluster = self.currently_tested_n_cluster
            SceneSetup.def_folder = self.def_folder \
                + 'k' + str(self.currently_tested_n_cluster) + '/d_' + str(i) + '/'

            SimSetup.tmax = self.currently_tested_simulation_time

            # Reset param that is connected to one another
            SceneSetup.reset_param()
            SimSetup.reset_param()
            

            self.set_directory(SceneSetup.def_folder)
            # Run the simulation
            self.single_loop(i)
            # Here I am assuming python do the clearing unused variable before the next simulation
            # If the code tends to slow down, manual clearing might be needed

        print('FINISHED running for ' + str(self.currently_tested_n_cluster) + ' number of clusters, for ' \
              + str(self.currently_tested_simulation_time) + ' seconds each.\n' )
    

    @staticmethod
    def set_directory(dir): # check and generate directory if needed
        if not os.path.isdir(dir):
            os.makedirs(dir)
            print("created folder : ", dir)


    @staticmethod
    def single_loop(iter):
        print("start loop no: " + str(iter))    

        # Ìnitialize Simulation
        sim = Simulate()
        # Step through simulation
        for i in range(round(SimSetup.tmax/SimSetup.Ts)+2): sim.loop_sequence(i)

        print("Finish loop no: " + str(iter))    
        print('------------------------------------------\n\n\n')
        # TODO: logger and profiling


    @staticmethod
    def process_batch_K_sim_pkl(default_path, total_iteration, n_cluster):
        # ---------------------------------------------------
        from scenarios_SI.AgriJournal_pickleplot import extract_single_sim_data

        print('Processing folder (SIMULATION) ' + default_path + \
              ' for ' + str(total_iteration) + 'iteration data' \
              ' and k=' + str(n_cluster))

        # Initialize Places to store data
        holes_cov_ratio = None
        final_flight_dist = None
        data_len_time = 0

        # Loop over each monte-carlo data
        for data_num in range(total_iteration):
            # Grab data in each iteration
            # ---------------------------------------
            f_name_read = default_path + '/k' + str(n_cluster) + '/d_' + str(data_num) + '/sim_data.pkl'    

            # Extract important data from pickle
            time_data, \
                dict_total_cov_ratio_each_cluster, total_cov_ratio_all_cluster, \
                dict_holes_cov_ratio_each_cluster, holes_cov_ratio_all_cluster, \
                flight_dist, final_distance, furthest_distance \
                = extract_single_sim_data(f_name_read, n_cluster, SceneSetup.robot_num)

            # Initialize size if it is the first time
            if holes_cov_ratio is None:
                data_len_time = holes_cov_ratio_all_cluster.size
                holes_cov_ratio = np.zeros((total_iteration, data_len_time))
                final_flight_dist = np.zeros((total_iteration, SceneSetup.robot_num))
            # Fill each row of data
            holes_cov_ratio[data_num] = holes_cov_ratio_all_cluster
            final_flight_dist[data_num] = np.array(final_distance)            

        # Compute coverage ratio mean and standard deviation on each time step
        holes_cov_ratio_mean = np.zeros( data_len_time )
        holes_cov_ratio_std = np.zeros( data_len_time )
        for i in range(data_len_time):
            holes_cov_ratio_mean[i] = np.mean(holes_cov_ratio[:,i])
            holes_cov_ratio_std[i] = np.std(holes_cov_ratio[:,i])
        
        final_flight_dist_mean = np.zeros( SceneSetup.robot_num )
        final_flight_dist_std = np.zeros( SceneSetup.robot_num )
        # Compute final flight distance mean and standard deviation on each robot
        for i in range(SceneSetup.robot_num):
            final_flight_dist_mean[i] = np.mean(final_flight_dist[:,i])
            final_flight_dist_std[i] = np.std(final_flight_dist[:,i])

        return time_data, \
            holes_cov_ratio_mean, holes_cov_ratio_std, \
            final_flight_dist_mean, final_flight_dist_std


    @staticmethod
    def process_batch_K_task_pkl(default_path, total_iteration, n_cluster, sr_area):

        print('Processing folder (TASK-ALLOCATION) ' + default_path + \
              ' for ' + str(total_iteration) + 'iteration data' \
              ' and k=' + str(n_cluster))

        saved_task_area = None
        eval_allocate = np.zeros((total_iteration, n_cluster))

        robot_num = 10
        def_min_dist = np.zeros((total_iteration, robot_num))
        eval_distance = np.zeros((total_iteration, robot_num))

        # Loop over each monte-carlo data
        for data_num in range(total_iteration):
            # Grab data in each iteration
            # ---------------------------------------
            f_name_read = default_path + '/k' + str(n_cluster) + '/d_' + str(data_num) + '/ACO_allocation.pkl'    

            with open(f_name_read, 'rb') as f:
                dict_centroid, dict_dividedPoints, dict_boundingBox, chosenAgentIdx, dict_task_area = pickle.load(f)
                # print('Finished loading data of Task Allocation in ' + f_name_read)
            
            if saved_task_area is None: saved_task_area = dict_task_area

            for c_id in range(n_cluster):
                sensing_allocation = 0.
                for s_id in chosenAgentIdx[c_id]: sensing_allocation += sr_area[s_id]
                # Allocation metrix is --> ROI size - sum of assigned drone's sensing size / ROI size
                # val = (dict_task_area[c_id] - sensing_allocation)/dict_task_area[c_id]
                # eval_allocate[data_num, c_id] = np.max([0, val])
                # Allocation metrix is --> sum of assigned drone's sensing size / ROI size
                val = sensing_allocation /dict_task_area[c_id]
                eval_allocate[data_num, c_id] = np.min([1, val])

            # Grab SIMULATION data in each iteration
            # ---------------------------------------
            # This part is needed because we don't save initial position in the beginning
            f_sim_read = default_path + '/k' + str(n_cluster) + '/d_' + str(data_num) + '/sim_data.pkl'    
            with open(f_sim_read, 'rb') as fs: simData = pickle.load(fs)
            init_pos = np.zeros((robot_num, 2))
            for i in range(robot_num):
                # Get each drone initial position
                init_pos[i,0] = simData['stored_data']['pos_x_' + str(i)][0]
                init_pos[i,1] = simData['stored_data']['pos_y_' + str(i)][0]

                # Minimum distance to any centroid
                diff_dist = dict_centroid - init_pos[i]
                def_min_dist[data_num, i] = min(np.linalg.norm(diff_dist, axis=1))

            # Compare init position to the allocated cluster
            for c_id in range(n_cluster):
                cluster_cent = dict_centroid[c_id]
                # Allocation metrix is --> drone's distance to the ROI's centroid
                for s_id in chosenAgentIdx[c_id]: 
                    diff = init_pos[s_id] - cluster_cent
                    eval_distance[data_num, s_id] = np.linalg.norm(diff)

        # Compute mean and std for eval_allocate
        eval_allocate_mean = np.zeros( n_cluster )
        eval_allocate_std = np.zeros( n_cluster )
        for i in range( n_cluster ):
            eval_allocate_mean[i] = np.mean(eval_allocate[:,i])
            eval_allocate_std[i] = np.std(eval_allocate[:,i])

        # Allocation metrix is --> deviation of 
        # drone's distance to the ROI's centroid 
        # vs the true minimum
        eval_deviate = eval_distance - def_min_dist

        # Compute mean and std for eval_distance
        eval_distance_mean = np.zeros( robot_num )
        eval_distance_std = np.zeros( robot_num )
        eval_deviate_mean = np.zeros( robot_num )
        eval_deviate_std = np.zeros( robot_num )
        for i in range( robot_num ):
            eval_distance_mean[i] = np.mean(eval_distance[:,i])
            eval_distance_std[i] = np.std(eval_distance[:,i])
            eval_deviate_mean[i] = np.mean(eval_deviate[:,i])
            eval_deviate_std[i] = np.std(eval_deviate[:,i])

        return saved_task_area, eval_allocate_mean, eval_allocate_std, \
            eval_distance_mean, eval_distance_std, \
            eval_deviate_mean, eval_deviate_std


    def post_process_data(self, n_cluster, show_plot=True, save_plot=True):
        # ---------------------------------------------------
        # DEFAULT SETTINGS
        default_path = self.def_folder
        total_iteration = self.total_batch

        # # IF SMALLER TESTINGS IS NEEDED
        # default_path = ''
        # total_iteration = self.iter_per_batch

        time_data, holes_cov_ratio_mean, holes_cov_ratio_std, \
            final_flight_dist_mean, final_flight_dist_std \
            = self.process_batch_K_sim_pkl(default_path, total_iteration, n_cluster)

        # Display Processed data: final flight distance
        # ---------------------------------------
        print('------------------------------------------\n\n\n')
        print('Aggregated results for the final flight distance')
        print('Mean', final_flight_dist_mean)
        print('Standard Deviation', final_flight_dist_std)

        # PLOT Processed data: coverage ratio
        # ---------------------------------------
        FS = 14 # font size

        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True

        ax.plot(time_data, holes_cov_ratio_mean, color = 'k', label='all')
        plt.fill_between(time_data, holes_cov_ratio_mean - holes_cov_ratio_std, holes_cov_ratio_mean + holes_cov_ratio_std, alpha=0.2)
        ax.set(xlabel="t [s]", ylabel="% uncovered broken sensor")
        ax.grid()
        plt.ylim(-0.1, 1.1)
        # plt.legend(loc='best')
        if save_plot:
            figname = default_path + '/k' + str(n_cluster) + '_monte_carlo_coverage_hole_ratio.pdf'
            plt.savefig(figname, bbox_inches="tight", dpi=300)
            print( 'export figure: ' + figname, flush=True)
        if show_plot: plt.show()

        # pass the information so that it can be used later in the comparison with other k_value
        return holes_cov_ratio_mean, holes_cov_ratio_std, \
            final_flight_dist_mean, final_flight_dist_std


    def post_process_data_all(self):
        
        total_cluster = 6
        robot_num = 10

        # Initialize matrix to store the data
        # row vs column is number of roi vs each drone
        array_flight_distance_mean = np.zeros((total_cluster, robot_num))
        array_flight_distance_std = np.zeros((total_cluster, robot_num))

        for i in range(total_cluster):
            holes_cov_ratio_mean, holes_cov_ratio_std, \
                final_flight_dist_mean, final_flight_dist_std \
                = self.post_process_data(i+1, show_plot=False, save_plot=True)

            # # dummy data
            # final_flight_dist_mean = np.array( [i+1]*robot_num )
            # final_flight_dist_std = np.array( [i/10]*robot_num )

            array_flight_distance_mean[i] = final_flight_dist_mean
            array_flight_distance_std[i] = final_flight_dist_std

        # PLOT Processed data: flight time barchart
        # ---------------------------------------
        FS = 14 # font size

        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True

        X_axis = np.arange(robot_num)
        width, margin = 0.1, 0.11
        spread = np.array( list(range(robot_num)) ) - ((robot_num-1)/2)

        for i in range(robot_num):
            plt.bar(X_axis + spread[i]*margin, array_flight_distance_mean[:,i], width, \
                    yerr=array_flight_distance_std[:,i], capsize=2, label = 'Quadrotor '+str(i+1))
  
        plt.xticks(X_axis, ['1','2','3','4','5','6','7','8','9','10'])
        # plt.ylim((0, 120))
        plt.xlabel("Number of ROIs")
        plt.ylabel("Flight Distances [m]")
        # plt.title("Flight Distance Performance")
        plt.legend()
        save_plot, show_plot = True, False
        if save_plot:
            figname = self.def_folder + 'monte_carlo_flight_distances_barchart.pdf'
            plt.savefig(figname, bbox_inches="tight", dpi=300)
            print( 'export figure: ' + figname, flush=True)
        if show_plot: plt.show()

