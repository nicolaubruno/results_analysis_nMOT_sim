#
# Libraries and modules
#--
import sys, os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq, curve_fit
from scipy.special import erf
#--

#
class Results:

    #--
    # Attributes 
    #--

    # Constants in the international system
    @property
    def ctes(self):
        return self._ctes
 
    # (Series)
    @property
    def atom(self):
        return self._atom

    # (Series)
    @property
    def transition(self):
        return self._transition

    # (Series) Initial conditions
    @property
    def ini_conds(self):
        return self._ini_conds

    # (Series) Performance
    @property
    def perform(self):
        return self._perform

    # (Series) Magnetic field
    @property
    def B_params(self):
        return self._B_params

    # Zeeman slower
    @property
    def zeeman_slower(self):
        return self._zeeman_slower

    # (Dictionary)
    @property
    def beams(self):
        return self._beams

    # (Dataframe) Information about the parameters
    @property
    def info(self):
        return self._info

    # Identification
    @property
    def id(self):
        return self._id

    # 3D-Histogram of positions
    @property
    def pos_3Dhist(self):
        return self._pos_3Dhist
    
    # Marginal histograms of positions
    @property
    def pos_hist(self):
        return self._pos_hist

    # 3D-Histogram of velocities
    @property
    def vel_3Dhist(self):
        return self._vel_3Dhist
    
    # Marginal histograms of velocities
    @property
    def vel_hist(self):
        return self._vel_hist
    
    # Trapped atoms
    @property
    def trapped_atoms(self):
        return self._trapped_atoms

    # Escape velocity
    @property
    def escape_vel(self):
        return self._escape_vel

    # Average escape time
    @property
    def average_escape_time(self):
        return self._average_escape_time

    # Looping status
    @property
    def loop(self):
        return self._loop

    # Directories
    @property
    def dirs(self):
        return self._dirs

    #
    @property
    def cut_trap_depth(self):
        return self._cut_trap_depth
    
    #--
    # Operational Methods 
    #--

    #
    def __init__(self, code, group, data_dir, subgroup = None, shortname = None, is_new = False):
        # Constants (SI)
        self._ctes = {
            'u': 1.660539040e-27,\
            'k_B': 1.38064852e-23,\
            'hbar': 1.0544718e-34,\
            'h': 6.626070040e-34
        }

        # Cut reference for the trap depth calculation
        self._cut_trap_depth = 0.5

        # Loop variable
        self._loop = {
            "var": None,\
            "values":[],\
            "length": 0,\
            "active": 0
        }

        # Directories
        self._dirs = {
            "model":data_dir,\
            "root":data_dir,\
        }

        # Identification
        #--
        self._id = {}
        self._id["code"] = int(code)
        self._id["name"] = shortname
        self._id["group"] = group
        self._id["subgroup"] = subgroup
        #--

        # Create a blank result object
        if is_new: 
            self.__new()

        # Get an existing results set
        else:
            # Get name
            self.__get_name()

        # Add directories
        #--
        self._dirs["group"] = self.dirs["root"] + "group_" + self.id["group"] + '/'
        if self.id["subgroup"] is not None: self._dirs["group"] += "subgroup_" + self.id["subgroup"] + '/'

        self._dirs["result"] = self.dirs["group"] + str(self.id["code"])
        if self.id["name"] is not None: self._dirs["result"] += "_" + self.id["name"]
        self._dirs["result"] += "/"
        #--

        # Get loop
        if not is_new: self.__get_loop()
        self.loop_idx(0)

    #
    def __get_name(self):
        # Status of result directory
        is_res_dir_exists = False

        # Result directory
        res_dir_path = self.dirs["root"]
        res_dir_path += "group_" + self.id["group"] + "/"
        if self.id["subgroup"] is not None:
            res_dir_path += "subgroup_" + self.id["subgroup"] + "/"

        # Get results directory
        res_dir_pter = os.scandir(res_dir_path)
        for res_dir in res_dir_pter:
            splited_res_dir_name = res_dir.name.split("_")

            if int(splited_res_dir_name[0]) == self.id["code"]:
                self._id["name"] = "_".join(splited_res_dir_name[1:])
                is_res_dir_exists = True
                break

        # Check result directory
        if not is_res_dir_exists:
            raise ValueError("Result directory does not exist")
    
    # Get attributes
    def __get_attr(self):
        # Directory of parameters
        params_dir = self.dirs["active_res"] + "parameters/"

        # Atom
        path = params_dir + "atom.csv"
        self._atom = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Transition
        path = params_dir + "transitions.csv"
        self._transition = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Beams
        #--
        self._beams = {'main': None, 'setup': None, 'sidebands': None}

        # Main
        path = params_dir + "beams/general.csv"
        self._beams['main'] = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Setup
        path = params_dir + "beams/setup.csv"
        self._beams['setup'] = pd.read_csv(path, header=0)
        self._beams['setup'].index += 1

        # Sidebands
        #path = params_dir + "beams/sidebands.csv"
        #self._beams['sidebands'] = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)
        #--

        # Initial conditions
        path = params_dir + "initial_conditions.csv"
        self._ini_conds = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Performance
        path = params_dir + "performance.csv"
        self._perform = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Magnetic field
        path = params_dir + "magnetic_field.csv"
        self._B_params = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Zeeman slower
        #--
        #self._zeeman_slower = {"beam": None, "transition": None}

        # Beam
        #path = params_dir + "zeeman_slower/beam.csv"
        #self._zeeman_slower['beam'] = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)

        # Transition
        #path = params_dir + "zeeman_slower/transition.csv"
        #self._zeeman_slower['transition'] = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)
        #--

        # Information about the parameters
        path = self._dirs["model"] + "parameters/informations.csv"
        self._info = pd.read_csv(path, header=0)
        self._info.set_index("parameter", inplace=True)
        self._info.fillna("", inplace=True)

    # Get distributions
    def __get_dists(self):
        #
        # 3D-Histograms
        #--
        self._pos_3Dhist = {
            'freqs': None,\
            'dens': None,\
            'bins': None
        }


        self._vel_3Dhist = {
            'freqs': None,\
            'dens': None,\
            'bins': None
        }
        #--

        # Marginal histograms
        self._pos_hist = [{"freqs":[], "dens":[], "bins":[]} for i in range(3)]
        self._vel_hist = [{"freqs":[], "dens":[], "bins":[]} for i in range(3)]

        #
        # 3D-Histograms of positions
        #--
        path = self.dirs["active_res"] + 'positions.csv'
        if os.path.exists(path):
            #
            # Read histogram file
            self._pos_3Dhist["freqs"] = np.array(pd.read_csv(path, index_col=0).squeeze()).reshape((int(self.perform['num_bins']), int(self.perform['num_bins']), int(self.perform['num_bins'])))

            #
            # Filter frequencies considering the waist size as a threshold

            #
            # Densities
            self._pos_3Dhist["dens"] = self.pos_3Dhist["freqs"] / np.sum(self.pos_3Dhist["freqs"])
            
            #
            # Bins
            self._pos_3Dhist["bins"] = np.zeros((3, int(self.perform['num_bins']))) - float(self.perform['max_r'])
            
            for i in range(3):
                for j in range(int(self.perform['num_bins'])):
                    delta = 2*float(self.perform['max_r']) / float(int(self.perform['num_bins']))
                    self._pos_3Dhist["bins"][i][j] += j*delta

            #
            # Marginal frequencies
            self._pos_hist[0]["freqs"] = np.sum(self.pos_3Dhist["freqs"], axis=(1, 2))
            self._pos_hist[1]["freqs"] = np.sum(self.pos_3Dhist["freqs"], axis=(0, 2))
            self._pos_hist[2]["freqs"] = np.sum(self.pos_3Dhist["freqs"], axis=(0, 1))

            #
            # Defined marginals
            for i in range(3):
                #
                # Marginal densities
                self._pos_hist[i]["dens"] = self._pos_hist[i]["freqs"] / np.sum(self._pos_hist[i]["freqs"])

                #
                # Marginal bins
                self._pos_hist[i]["bins"] = - np.ones(int(self.perform['num_bins'])) * float(self.perform['max_r'])
                delta = 2*float(self.perform['max_r']) / float(int(self.perform['num_bins']))

                for j in range(int(self.perform['num_bins'])):
                    self._pos_hist[i]["bins"][j] += j*delta
        #--

        #
        # 3D-Histograms of velocities
        #--
        path = self.dirs["active_res"] + 'velocities.csv'
        if os.path.exists(path):
            #
            # Read histogram file
            self._vel_3Dhist["freqs"] = np.array(pd.read_csv(path, index_col=0).squeeze()).reshape((int(self.perform['num_bins']), int(self.perform['num_bins']), int(self.perform['num_bins'])))

            #
            # Filter frequencies considering the waist size as a threshold

            #
            # Densities
            self._vel_3Dhist["dens"] = self.vel_3Dhist["freqs"] / np.sum(self.vel_3Dhist["freqs"])
            
            #
            # Bins
            self._vel_3Dhist["bins"] = np.zeros((3, int(self.perform['num_bins']))) - float(self.perform['max_v'])
            
            for i in range(3):
                for j in range(int(self.perform['num_bins'])):
                    delta = 2*float(self.perform['max_r']) / float(int(self.perform['num_bins']))
                    self._vel_3Dhist["bins"][i][j] += j*delta

            #
            # Marginal frequencies
            self._vel_hist[0]["freqs"] = np.sum(self.vel_3Dhist["freqs"], axis=(1, 2))
            self._vel_hist[1]["freqs"] = np.sum(self.vel_3Dhist["freqs"], axis=(0, 2))
            self._vel_hist[2]["freqs"] = np.sum(self.vel_3Dhist["freqs"], axis=(0, 1))

            #
            # Defined marginals
            for i in range(3):
                #
                # Marginal densities
                self._vel_hist[i]["dens"] = self._vel_hist[i]["freqs"] / np.sum(self._vel_hist[i]["freqs"])

                #
                # Marginal bins
                self._vel_hist[i]["bins"] = - np.ones(int(self.perform['num_bins'])) * float(self.perform['max_v'])
                delta = 2*float(self.perform['max_v']) / float(int(self.perform['num_bins']))

                for j in range(int(self.perform['num_bins'])):
                    self._vel_hist[i]["bins"][j] += j*delta
        #--

        #
        # Marginal histograms
        #--
        path = self.dirs["active_res"] + 'marginals.csv'
        if os.path.exists(path):
            #
            # Read file
            df = pd.read_csv(path, index_col=0)

            # Check if velocity exists
            check_vel = (('vx' in df.columns) and ('vy' in df.columns) and ('vz' in df.columns))

            #
            # Frequencies
            self._pos_hist[0]["freqs"] = np.array(df['x'])
            self._pos_hist[1]["freqs"] = np.array(df['y'])
            self._pos_hist[2]["freqs"] = np.array(df['z'])

            if check_vel:
                self._vel_hist[0]["freqs"] = np.array(df['vx'])
                self._vel_hist[1]["freqs"] = np.array(df['vy'])
                self._vel_hist[2]["freqs"] = np.array(df['vz'])

            #
            # Densities and bins of marginal histograms
            for i in range(3):
                # Densities
                self._pos_hist[i]["dens"] = self._pos_hist[i]["freqs"] / np.sum(self._pos_hist[i]["freqs"])
                if check_vel: self._vel_hist[i]["dens"] = self._vel_hist[i]["freqs"] / np.sum(self._vel_hist[i]["freqs"])

                #
                # Bins
                self._pos_hist[i]["bins"] = - np.ones(int(int(self.perform['num_bins']))) * float(self.perform['max_r'])
                if check_vel: self._vel_hist[i]["bins"] = - np.ones(int(int(self.perform['num_bins']))) * float(self.perform['max_v'])
                pos_delta = 2*float(self.perform['max_r']) / float(int(self.perform['num_bins']))
                if check_vel: vel_delta = 2*float(self.perform['max_v']) / float(int(self.perform['num_bins']))

                for j in range(int(self.perform['num_bins'])):
                    self._pos_hist[i]["bins"][j] += j*pos_delta
                    if check_vel: self._vel_hist[i]["bins"][j] += j*vel_delta
        #--

    #
    def __get_log(self):
        path = self.dirs["active_res"] + 'log.csv'
        if os.path.exists(path):
            log = pd.read_csv(path, header=0, index_col=0).squeeze().astype(object)
            
            # Trapped atoms
            if "trapped_atoms" in log:
                self._trapped_atoms = log['trapped_atoms']

            # Average escape time
            if "average_escape_time" in log: 
                self._average_escape_time = log['average_escape_time']

            # Escape velocity
            if "escape_vel" in log:
                self._escape_vel = log["escape_vel"]

    #
    def __get_loop_values(self, loop_str):
        # Return variable
        values = []

        # Check string
        if len(loop_str) > 4 and loop_str[0:4] == 'loop':
            #
            # Loop option 1
            if loop_str[4] == '[' and loop_str[-1] == ']':
                opts = loop_str[5:-1].split(' ')

                val = float(opts[0])
                end = float(opts[1])
                step = float(opts[2])
                
                if ((end - val) < 0 and step < 0) or ((end - val) > 0 and step > 0):
                    values = []

                    while val <= end:
                        values.append(val)
                        val += step
                else:
                    raise ValueError('Incorrect looping in the parameters')

            elif loop_str[4] == '{' and loop_str[-1] == '}':
                values = np.array(loop_str[5:-1].split(' '), dtype=float)
                values = sorted(values, key=(lambda x: float(x)))

            else:
                raise ValueError('Invalid loop variable')

        return values
    
    #
    def __get_loop(self):
        # Sorting
        pos = []

        # Check directories
        res_dir_pter = os.scandir(self.dirs["result"])
        for res_dir in res_dir_pter:
            if self.loop["length"] == 0: 
                loop_var = res_dir.name.split("_")
                if len(loop_var) > 1: 
                    self._loop["var"] = "_".join(loop_var[1:])

                else: break

            if self.loop["var"] is not None:
                self.loop_idx(self.loop["length"])
                self.loop["length"] += 1
                pos.append(int(res_dir.name.split("_")[0][3:]))

                # Magnetic field
                #--
                prohibited_variables = ["B_axial", "B_bias", "B_lin_grad"]

                if self.loop["var"] in prohibited_variables:
                    self._loop["var"] = None

                elif self.loop["var"] in self.B_params.index:
                    param = pd.read_csv(res_dir.path + "/parameters/magnetic_field.csv", header=0, index_col=0).squeeze().astype(object)
                    self._loop["values"].append(param[self.loop["var"]])
                #--

                # Initial conditions
                #--
                prohibited_variables = ["v_0_dir"]
                if self.loop["var"] in self.ini_conds.index:
                    param = pd.read_csv(res_dir.path + "/parameters/initial_conditions.csv", header=0, index_col=0).squeeze().astype(object)
                    self._loop["values"].append(param[self.loop["var"]])
                #--

                # Performance
                #--
                if self.loop["var"] in self.perform.index:
                    param = pd.read_csv(res_dir.path + "/parameters/performance.csv", header=0, index_col=0).squeeze().astype(object)
                    self._loop["values"].append(param[self.loop["var"]])
                #--

                # Atom
                #--
                prohibited_variables = ["symbol"]

                if self.loop["var"] in prohibited_variables:
                    self._loop["var"] = None

                elif self.loop["var"] in self.atom.index:
                    param = pd.read_csv(res_dir.path + "/parameters/atom.csv", header=0, index_col=0).squeeze().astype(object)
                    self._loop["values"].append(param[self.loop["var"]])
                #--

                # Transition
                #--
                if self.loop["var"] in self.transition.index:
                    param = pd.read_csv(res_dir.path + "/parameters/transition.csv", header=0, index_col=0).squeeze().astype(object)
                    self._loop["values"].append(param[self.loop["var"]])
                #--

                # Beams (main)
                #--
                if self.loop["var"] in self.beams['main'].index:
                    param = pd.read_csv(res_dir.path + "/parameters/beams/general.csv", header=0, index_col=0).squeeze().astype(object)
                    self._loop["values"].append(param[self.loop["var"]])
                #--

                # Beams (sidebands)
                #--
                #if self.loop["var"] in self.beams['sidebands'].index:
                #    param = pd.read_csv(res_dir.path + "/parameters/beams/sidebands.csv", header=0, index_col=0).squeeze().astype(object)
                #    self._loop["values"].append(param[self.loop["var"]])
                #--   

                # Zeeman Slower (beam)
                #--
                #if self.loop["var"] in self.zeeman_slower['beam'].index:
                #    param = pd.read_csv(res_dir.path + "/parameters/zeeman_slower/beam.csv", header=0, index_col=0).squeeze().astype(object)
                #    self._loop["values"].append(param[self.loop["var"]])
                #--     

                # Zeeman Slower (transition)
                #--
                #if self.loop["var"] in self.zeeman_slower['transition'].index:
                #    param = pd.read_csv(res_dir.path + "/parameters/zeeman_slower/transition.csv", header=0, index_col=0).squeeze().astype(object)
                #    self._loop["values"].append(param[self.loop["var"]])
                #--     
            else: break


        # Sort values
        loop_values = np.array(self.loop["values"])
        for i in range(self.loop["length"]):
            self.loop["values"][pos[i]-1] = loop_values[i]

    #
    def __new(self):
        # Create results directory
        #--
        dir_path = self.dirs["root"] + "group_" + self.id["group"] + "/"
        if self.id["subgroup"] is not None: dir_path += "subgroup_" + self.id["subgroup"] + "/"
        dir_path += str(self.id["code"])
        if self.id["name"] is not None: dir_path += '_' + self.id["name"]
        dir_path += '/'

        if os.path.exists(dir_path):
            raise ValueError("Results already exists!")

        else: os.mkdir(dir_path)
        #--

        # Create directories for each looping
        #--
        # Create new attributes     
        self.__create_attr()

        # Looping
        num_res = self.loop["length"] if self.loop["length"] > 0 else 1
        for i in range(num_res):
            # Result directory
            if self.loop["var"] is not None:
                res_dir = dir_path + "res" + str(i+1) + '_' + self.loop["var"] + '/'

            else:
                res_dir = dir_path + "res1/"

            # Create directory
            os.mkdir(res_dir)

            # Save parameters of the simulation
            #--
            params_dir = res_dir + "parameters/"
            os.mkdir(params_dir)
            os.mkdir(params_dir + "beams/")
            os.mkdir(params_dir + "zeeman_slower/")

            # Change loop variable 
            if self.loop["var"] is not None:
                self.loop["active"] = i
                self.__set_loop()

            self.atom.to_csv(params_dir + "atom.csv", header="atom")
            self.transition.to_csv(params_dir + "transition.csv", header="transition")
            self.ini_conds.to_csv(params_dir + "initial_conditions.csv", header="initial_conditions")
            self.perform.to_csv(params_dir + "performance.csv", header="performance")
            self.B_params.to_csv(params_dir + "magnetic_field.csv", header="magnetic_field")

            self.beams['main'].to_csv(params_dir + "beams/main.csv", header="beams_main")
            self.beams['setup'].to_csv(params_dir + "beams/setup.csv", index=False)
            self.beams['sidebands'].to_csv(params_dir + "beams/sidebands.csv", header="beams_sidebands")

            self.zeeman_slower["beam"].to_csv(params_dir + "zeeman_slower/beam.csv", header="beam")
            self.zeeman_slower["transition"].to_csv(params_dir + "zeeman_slower/transition.csv", header="ZS_transition")
            #--

    #
    def __create_attr(self):
        # Parameters directory
        self._dirs["active_res"] = self._dirs["model"]

        # Get attribute
        self.__get_attr()

        # Check looping values
        self.__set_loop(first_time = True)

    # Cast values of the parameters
    def __cast_params_values(self):
        # Atom
        self._atom['Z'] = int(self.atom['Z'])
        self._atom['mass'] = float(self.atom['mass'])

        # Transition
        self._transition['gamma'] = float(self.transition['gamma'])
        self._transition['lambda'] = float(self.transition['lambda'])
        self._transition['g_gnd'] = float(self.transition['g_gnd'])
        self._transition['g_exc'] = float(self.transition['g_exc'])
        self._transition['J_gnd'] = int(self.transition['J_gnd'])
        self._transition['J_exc'] = int(self.transition['J_exc'])

        # Magnetic field
        self._B_params['B_0'] = float(self.B_params['B_0'])

        # Initial conditions
        self._ini_conds['T_0'] = float(self.ini_conds['T_0'])
        self._ini_conds['v_0'] = float(self.ini_conds['v_0'])
        self._ini_conds['g_bool'] = int(self.ini_conds['g_bool'])

        # Performance
        self._perform['max_time'] = float(self.perform['wait_time']) + float(self.perform['recording_time'])
        self._perform['wait_time'] = float(self.perform['wait_time'])
        self._perform['dt'] = float(self.perform['dt'])
        self._perform['max_r'] = float(self.perform['max_r'])
        self._perform['max_v'] = float(self.perform['max_v'])
        self._perform['num_sim'] = int(self.perform['num_sim'])
        self._perform['num_bins'] = int(self.perform['num_bins'])
        self._perform['parallel_tasks'] = int(self.perform['parallel_tasks'])

        # Beams (main)
        self._beams['main']['delta'] = float(self.beams['main']['delta'])
        self._beams['main']['s_0'] = float(self.beams['main']['s_0'])
        self._beams['main']['w'] = float(self.beams['main']['w'])

        # Beams (sidebands)
        #self._beams['sidebands']['freq'] = float(self.beams['sidebands']['freq'])

        # Zeeman slower (beam)
        #self._zeeman_slower["beam"]['delta'] = float(self.zeeman_slower["beam"]['delta'])
        #self._zeeman_slower["beam"]['s_0'] = float(self.zeeman_slower["beam"]['s_0'])
        #self._zeeman_slower["beam"]['w'] = float(self.zeeman_slower["beam"]['w'])

        # Zeeman slower (transition)
        #self._zeeman_slower["transition"]['gamma'] = float(self.zeeman_slower["transition"]['gamma'])
        #self._zeeman_slower["transition"]['lambda'] = float(self.zeeman_slower["transition"]['lambda'])
        #self._zeeman_slower["transition"]['J_gnd'] = int(self.zeeman_slower["transition"]['J_gnd'])
        #self._zeeman_slower["transition"]['J_exc'] = int(self.zeeman_slower["transition"]['J_exc'])
        #self._zeeman_slower["transition"]['g_gnd'] = float(self.zeeman_slower["transition"]['g_gnd'])
        #self._zeeman_slower["transition"]['g_exc'] = float(self.zeeman_slower["transition"]['g_exc'])

    # Check if name exists
    def __check_name(self, name):
        #
        # Variables
        obj_scandir = os.scandir(self.root_dir)
        ret = False

        for path in obj_scandir:
            str_splited = path.name.split("_")
            code = int(str_splited[0])

            check_name = ""
            for j in range(1, len(str_splited)):
                if j == 1: check_name += str_splited[j]
                else: check_name += '_' + str_splited[j]

            if code == self.code:
                if check_name == name:
                    ret = True
                    break

        return ret  


    # Set loop values
    def __set_loop(self, first_time = False):
        # First time setting the loop
        if first_time:            
            # Magnetic field
            #--
            prohibited_variables = ["B_axial", "B_bias", "B_lin_grad"]

            for idx in self.B_params.index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.B_params[idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Initial conditions
            #--
            prohibited_variables = ["v_0_dir"]

            if first_time:
                for idx in self.ini_conds.index:
                    if not (idx in prohibited_variables):
                        values = self.__get_loop_values(str(self.ini_conds[idx]))
                        if len(values) > 0:
                            self._loop["var"] = idx
                            self._loop["values"] = values
                            self._loop["length"] = len(values)
                            break
            #--

            # Performance
            #--
            prohibited_variables = []

            for idx in self.perform.index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.perform[idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Atom
            #--
            prohibited_variables = ["symbol"]

            for idx in self.atom.index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.atom[idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Transition
            #--
            prohibited_variables = []

            for idx in self.transition.index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.transition[idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Beams (main)
            #--
            prohibited_variables = []

            for idx in self.beams['main'].index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.beams['main'][idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Beams (sidebands)
            #--
            prohibited_variables = []

            for idx in self.beams['sidebands'].index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.beams['sidebands'][idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Zeeman slower (beam)
            #--
            prohibited_variables = ["k", "pol"]

            for idx in self.zeeman_slower["beam"].index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.zeeman_slower["beam"][idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

            # Zeeman slower (transition)
            #--
            prohibited_variables = []

            for idx in self.zeeman_slower["transition"].index:
                if not (idx in prohibited_variables):
                    values = self.__get_loop_values(str(self.zeeman_slower["transition"][idx]))
                    if len(values) > 0:
                        self._loop["var"] = idx
                        self._loop["values"] = values
                        self._loop["length"] = len(values)
                        break
            #--

        # Change loop value
        #--
        if self.loop["var"] in self.B_params.index:
            self._B_params[self.loop["var"]] = self.loop["values"][self.loop["active"]]

        elif self.loop["var"] in self.ini_conds.index:
            self._ini_conds[self.loop["var"]] = self.loop["values"][self.loop["active"]]
        
        elif self.loop["var"] in self.perform.index:
            self._perform[self.loop["var"]] = self.loop["values"][self.loop["active"]]

        elif self.loop["var"] in self.atom.index:
            self._atom[self.loop["var"]] = self.loop["values"][self.loop["active"]]
        
        elif self.loop["var"] in self.transition.index:
            self._transition[self.loop["var"]] = self.loop["values"][self.loop["active"]]

        elif self.loop["var"] in self.beams['main'].index:
            self._beams['main'][self.loop["var"]] = self.loop["values"][self.loop["active"]]

        elif self.loop["var"] in self.beams['sidebands'].index:
            self._beams['sidebands'][self.loop["var"]] = self.loop["values"][self.loop["active"]]

        elif self.loop["var"] in self.zeeman_slower['beam'].index:
            self._zeeman_slower["beam"][self.loop["var"]] = self.loop["values"][self.loop["active"]]

        elif self.loop["var"] in self.zeeman_slower['transition'].index:
            self._zeeman_slower["transition"][self.loop["var"]] = self.loop["values"][self.loop["active"]]
        #--

    #
    def loop_idx(self, idx):
        # Change active loop index
        self._loop["active"] = idx

        # Change directory
        self._dirs["active_res"] = self.dirs["result"] 
        self._dirs["active_res"] += "res" + str(self.loop["active"] + 1)
        
        if self.loop["var"] is None: 
            self._dirs["active_res"] += "/"
        else:
            self._dirs["active_res"] += "_" + self.loop["var"] + "/"

        self.__get_attr()
        self.__get_dists()
        self.__get_log()
        self.__cast_params_values()

    #--
    # Methods to save data
    #--

    # Add frequencies in the 3D histogram of positions
    def add_positions(self, pos_freqs_arr):
        # Transform the 3D-array in a 1D-array
        #--
        indexes = []
        values = []

        for i in range(int(self.perform['num_bins'])):
            for j in range(int(self.perform['num_bins'])):
                for k in range(int(self.perform['num_bins'])):
                    indexes.append("[%d,%d,%d]" % (i+1, j+1, k+1))
                    values.append(pos_freqs_arr[int(self.perform['num_bins'])**2 * i + int(self.perform['num_bins'])*j + k])

        values = np.array(values)
        #--

        # Save file
        path = self.dirs["active_res"] + "/positions.csv"
        pos_freqs = pd.Series(values, index=indexes).astype("int32")
        pos_freqs.fillna(0, inplace=True)
        pos_freqs.to_csv(path)

        # Update distributions
        self.__get_dists()
        self.__get_log()

        # Release memory
        del values
        del indexes
        del pos_freqs
        del path

        gc.collect()

    # Add frequencies in the 3D histogram of velocities
    def add_velocities(self, vel_freqs_arr):
        # Transform the 3D-array in a 1D-array
        #--
        indexes = []
        values = []

        for i in range(int(self.perform['num_bins'])):
            for j in range(int(self.perform['num_bins'])):
                for k in range(int(self.perform['num_bins'])):
                    indexes.append("[%d,%d,%d]" % (i+1, j+1, k+1))
                    values.append(vel_freqs_arr[int(self.perform['num_bins'])**2 * i + int(self.perform['num_bins'])*j + k])

        values = np.array(values)
        #--

        # Save file
        path = self.dirs["active_res"] + "/velocities.csv"
        vel_freqs = pd.Series(values, index=indexes).astype("int32")
        vel_freqs.fillna(0, inplace=True)
        vel_freqs.to_csv(path)

        # Update distributions
        self.__get_dists()
        self.__get_log()

        # Release memory
        del values
        del indexes
        del vel_freqs
        del path

        gc.collect()

    # Add frequencies in the marginal histograms of positions
    def add_marginals(self, pos_freqs_arr, vel_freqs_arr):
        data = {
            'x': pos_freqs_arr[0],\
            'y': pos_freqs_arr[1],\
            'z': pos_freqs_arr[2],\
            'vx': vel_freqs_arr[0],\
            'vy': vel_freqs_arr[1],\
            'vz': vel_freqs_arr[2]
        }

        path = self.dirs["active_res"] + "marginals.csv"
        freqs = pd.DataFrame(data).astype("int32")
        freqs.fillna(0, inplace=True)
        freqs.to_csv(path)

        #
        # Release memory
        del freqs
        del data
        del path

        gc.collect()

    # Add trapped atoms
    def add_infos(self, data):
        path = self.dirs["active_res"] + "log.csv"
        pd.Series(data, name="log").to_csv(path, header="log")

    #--
    # Methods to processing and view data 
    #--

    # Escape flux of atoms
    def escape_flux_atoms(self):
        data = 0

        # With looping
        if len(self.loop["var"]) > 0:
            Y = np.zeros(self.loop["length"])
            X = np.array(self.loop["values"], dtype="float")

            for i in range(self.loop["length"]):
                self.loop_idx(i)

                # Get escape flux of atoms
                if self.average_escape_time > 0:
                    Y[i] = 2*np.pi*self.transition["gamma"]*1e3 * (1 - self.trapped_atoms / self.perform["num_sim"]) / (self.average_escape_time)

            data = (X, Y)

        # Without looping
        elif self.average_escape_time > 0: 
            data = 2*np.pi*self.transition["gamma"]*1e3 * (1 - self.trapped_atoms / self.perform["num_sim"]) / self.average_escape_time

        return data

    # Normalized trapped atoms
    def normalized_trapped_atoms(self, pin_loop=False):
        # With looping
        #--
        if self.loop["length"] > 0 and (not pin_loop):
            Y = np.zeros(self.loop["length"])
            X = np.array(self.loop["values"], dtype="float")

            for i in range(self.loop["length"]):
                self.loop_idx(i)
                Y[i] = (self.trapped_atoms / self.perform['num_sim'])

            data = (X, Y)
        #--

        # Without loop
        #--
        else:
            data = self.trapped_atoms / self.perform['num_sim']
        #--

        return data

    # Get all escape velocities
    def all_escape_velocities(self):
        # Data
        data = None

        # With loop
        #--
        if self.loop["length"] > 0:
            X, Y, Z = [], [], []

            for i in range(self.loop["length"]):
                # Change loop
                self.loop_idx(i)

                # Check value of the escape velocity
                if self.escape_vel > 0:
                    X.append(float(self.loop["values"][i]))
                    Y.append(self.escape_vel)
                    Z.append(self.normalized_trapped_atoms(pin_loop=True))

            # Loop values
            X = np.array(X, dtype="float")

            # Escape velocities
            Y = np.array(Y, dtype="float")

            # Normalized trapped atoms
            Z = np.array(Z, dtype="float")

            # Data
            data = (X, Y, Z)
        #--

        # Without loop
        #--
        else: data = self.escape_vel
        #--

        return data

    #
    def centre_of_mass(self, axis=[0,1,2], pin_loop = False):
        # Data
        data = None

        # Returns the best parameters to fit a Gaussian function
        def fit_gaussian(x, y):
            #
            # Gaussian function
            def gaussian(x, mean, std_dev): \
                return np.max(y) * np.exp(-((x - mean)/std_dev)**2 / 2)

            #
            # Guess values
            #--
            guess_values = np.zeros(2)
            guess_values[0] = np.sum(x*y) / np.sum(y)
            guess_values[1] = np.sqrt(np.sum(y * (x - guess_values[0])**2) / np.sum(y))
            #--
            #--

            popt, pcov = curve_fit(gaussian, x, y, guess_values)

            return popt

        # Without looping
        #--
        if self.loop["length"] == 0 or pin_loop:
            # Variables
            if len(axis) == 1:
                r_0, std_r_0 = fit_gaussian(self.pos_hist[axis[0]]["bins"], self.pos_hist[axis[0]]["dens"])
            
            else:
                r_0 = np.zeros(len(axis))
                std_r_0 = np.zeros(len(axis))

                #
                # Fit a normal Gaussian function
                for idx, val in enumerate(axis):
                    r_0[idx], std_r_0[idx] = fit_gaussian(self.pos_hist[val]["bins"], self.pos_hist[val]["dens"])
        
            data = (r_0, std_r_0)
        #--

        # With looping
        #--
        else:
            #
            # Set initial variables
            #--
            if len(axis) == 1:
                r_0 = np.zeros(self.loop["length"])
                std_r_0 = np.zeros(self.loop["length"])
            
            else:
                r_0 = np.zeros((len(axis),self.loop["length"]))
                std_r_0 = np.zeros((len(axis),self.loop["length"]))

            X = np.array(self.loop["values"], dtype="float")
            #--

            # Mass centre for each looping value
            for i in range(self.loop["length"]):
                #
                # Get looping values
                self.loop_idx(i)

                if len(axis) > 1:
                    for j in range(len(axis)):
                        r_0[j][i], std_r_0[j][i] = fit_gaussian(self.pos_hist[axis[j]]["bins"], self.pos_hist[axis[j]]["dens"])

                else:
                    x = self.pos_hist[axis[0]]["bins"]
                    p = self.pos_hist[axis[0]]["dens"]

                    r_0[i], std_r_0[i] = fit_gaussian(self.pos_hist[axis[0]]["bins"], self.pos_hist[axis[0]]["dens"])

            data = (X, r_0, std_r_0)
        #--

        return data

    #
    def average_velocity(self, axis=[0,1,2], fixed_loop_idx = False):
        #
        # Returns the best parameters to fit a Gaussian function
        def fit_gaussian(x, y):
            # Convert to numpy array
            x = np.array(x)
            y = np.array(y)

            #
            # Gaussian function
            def gaussian(x, mean, std_dev):
                return np.max(y) * np.exp(-((x - mean)/std_dev)**2 / 2)

            #
            # Guess values
            #--
            guess_values = np.zeros(2)
            guess_values[0] = np.sum(x*y) / np.sum(y)
            guess_values[1] = np.sqrt(np.sum(y * (x - guess_values[0])**2) / np.sum(y))
            #--
            #--

            popt, pcov = curve_fit(gaussian, x, y, guess_values)

            return popt

        #
        # Without looping
        #--
        if len(self.loop["var"]) == 0 or fixed_loop_idx:
            # Variables
            if len(axis) == 1:
                vel_c, std_vel_c = fit_gaussian(self.vel_hist[axis[0]]["bins"], self.vel_hist[axis[0]]["dens"])
            
            else:
                vel_c = np.zeros(len(axis))
                std_vel_c = np.zeros(len(axis))

                #
                # Fit a normal Gaussian function
                for idx, val in enumerate(axis):
                    vel_c[idx], std_vel_c[idx] = fit_gaussian(self.vel_hist[val]["bins"], self.vel_hist[val]["dens"])
        #--

        #
        # With looping
        #--
        else:
            #
            # Set initial variables
            #--
            if len(axis) == 1:
                vel_c = np.zeros(len(self.loop["values"]))
                std_vel_c = np.zeros(len(self.loop["values"]))
            
            else:
                vel_c = np.zeros((len(axis),len(self.loop["values"])))
                std_vel_c = np.zeros((len(axis),len(self.loop["values"])))
            #--

            # Mass centre for each looping value
            for i in range(len(self.loop["values"])):
                #
                # Get looping values
                self.loop_idx(i)

                if len(axis) > 1:
                    for j in axis:
                        vel_c[j][i], std_vel_c[j][i] = fit_gaussian(self.vel_hist[j]["bins"], self.vel_hist[j]["dens"])

                else:
                    x = self.vel_hist[axis[0]]["bins"]
                    p = self.vel_hist[axis[0]]["dens"]

                    vel_c[i], std_vel_c[i] = fit_gaussian(self.vel_hist[axis[0]]["bins"], self.vel_hist[axis[0]]["dens"])
        #--

        return vel_c, std_vel_c      

    # Get temperatures in uK
    def temperature(self, fixed_loop_idx = False, method=0):
        #
        # Without looping
        #--
        if len(self.loop["var"]) == 0 or fixed_loop_idx:
            v_av, v_dev = self.average_velocity(axis=[0, 1, 2], fixed_loop_idx=fixed_loop_idx)
            
            if method == 0:
                temp = ((np.sum(v_dev*1e-2)/3)**2 * float(self.atom['mass']) * self.ctes['u']) / self.ctes['k_B']

            elif method == 1:
                v_av = v_av*1e-2
                v_var = (v_dev*1e-2)**2
                v_square = v_var + v_av**2

                temp = (np.sum(v_square) * float(self.atom['mass']) * self.ctes['u']) / (3*self.ctes['k_B'])
            
            else:
                raise ValueError("Invalid method")

        #--

        #
        # With looping
        #--
        else:
            v_av, v_dev = self.average_velocity(axis=[0, 1, 2])

            if method == 0:
                temp = ((np.sum(v_dev*1e-2, axis=0)/3)**2 * float(self.atom['mass']) * self.ctes['u']) / self.ctes['k_B']

            elif method == 1:
                v_av = v_av*1e-2
                v_var = (v_dev*1e-2)**2
                v_square = v_var + v_av**2

                temp = (np.sum(v_square, axis=0) * float(self.atom['mass']) * self.ctes['u']) / (3*self.ctes['k_B'])

        return temp   

    # Doppler temperature
    def doppler_temperature(self, power_broadening=False, fixed_loop_idx = False):
        if power_broadening:
            alpha = np.sqrt(1 + self.beams['main']['s_0'])
        else:
            alpha = 0;

        #
        # Check looping
        if self.loop["var"] == "gamma" and fixed_loop_idx:
            temp = np.zeros(len(self.loop["values"]))
            for i, gamma in enumerate(self.loop["values"]):
                temp[i] = 1e9*(self.ctes['hbar'] * gamma * alpha) / (2 * self.ctes['k_B']) # uK

        else:
            temp = 1e9*(self.ctes['h'] * self.transition['gamma'] * alpha) / (2 * self.ctes['k_B']) # uK

        return temp

    # Trapped atoms
    def all_trapped_atoms(self, fixed_loop_idx = False):
        #
        # Without looping
        #--
        if len(self.loop["var"]) == 0 or fixed_loop_idx:
            res = self.trapped_atoms

        #
        # With looping
        #--
        else:
            res = np.zeros(len(self.loop["values"]))

            for i in range(len(self.loop["values"])):
                self.loop_idx(i)
                res[i] = self.trapped_atoms
        #--

        return res

    # Capture velocity
    def capture_velocity(self, fit_func = "erf"):
        # Check loop variable
        if self.loop["var"] == 'v_0' and self.loop["length"] > 1:
            # Get velocities and trapped atoms ratio
            #--
            vel = np.array(self.loop['values'], dtype="float")
            ratio = np.zeros(self.loop['length'])

            for i, val in enumerate(self.loop["values"]):
                self.loop_idx(i)
                ratio[i] = self.trapped_atoms

            ratio = ratio / self.perform["num_sim"]
            max_vel = np.max(vel)
            min_vel = np.min(vel)
            #--

            # Get capture velocity
            #--
            # Polynomial fitting
            if fit_func == "poly":
                # Fit polynomial
                fit_params = np.polyfit(vel, ratio, 10, full = False, cov = False)

                # Polynomial function
                def f(x):
                    y = -self._cut_trap_depth

                    for i in range(11):
                        y += fit_params[10 - i]*x**i

                    return y

                # Get capture velocity
                if f(min_vel) > 0 and f(min_vel)*f(max_vel) < 0: 
                    vel_c = brentq(f, min_vel, max_vel, full_output = False)
                else: vel_c = -1

            # Erf function fitting
            elif fit_func == "erf":
                # General complementary error function
                def general_erfc(t, mean, std_dev, amp):
                    return amp*(1 - (erf((t - mean) / np.sqrt(2 * std_dev**2)) - erf((- mean) / np.sqrt(2 * std_dev**2))) / 2)

                # Get data
                fit_params, covs = curve_fit(general_erfc, vel, ratio, bounds=([min(vel), 0, 0], [max(vel), (max(vel) - min(vel)), 1]))
                f = lambda x: general_erfc(x, fit_params[0], fit_params[1], fit_params[2]) - self._cut_trap_depth
                if f(min_vel) > 0 and f(min_vel)*f(max_vel) < 0:
                    vel_c = brentq(f, min_vel, max_vel, full_output = False)
                else: vel_c = -1
            #--

        else: raise ValueError("The loop variable must be v_0 to calculate the capture velocity")

        return vel_c, fit_params

    # Capture temperature
    def capture_temperature(self, fit_func = "poly"):
        T_mean, T_std_dev = (0,0)

        if self.loop["var"] == 'T_0' and len(self.loop["values"]) > 1:
            T = self.loop['values']
            ratio = np.zeros(len(self.loop['values']))

            for i, val in enumerate(self.loop["values"]):
                self.loop_idx(i)
                ratio[i] = self.trapped_atoms

            ratio = ratio / self.perform['num_sim']

            # Get capture temperature
            #--
            # Polynomial fitting
            if fit_func == "poly":
                # Fit polynomial
                fit_params = np.polyfit(T, ratio, 10, full = False, cov = False)

                # Polynomial function
                def f(x):
                    y = -self._cut_trap_depth

                    for i in range(11):
                        y += fit_params[10 - i]*x**i

                    return y

                # Get capture velocity
                max_T = np.max(T)
                min_T = np.min(T)
                if f(min_T) > 0 and f(min_T)*f(max_T) < 0: T_c = brentq(f, min_T, max_T, full_output = False)
                else: T_c = -1

            # Erf function fitting
            elif fit_func == "erf":
                # General complementary error function
                def general_erfc(t, mean, std_dev):
                    return 1 - (erf((t - mean) / np.sqrt(2 * std_dev**2)) - erf((- mean) / np.sqrt(2 * std_dev**2))) / 2

                # Get data
                fit_params, covs = curve_fit(general_erfc, T, ratio, bounds=([min(T), 0], [max(T), (max(T) - min(T))]))
                f = lambda x: general_erfc(x, fit_params[0], fit_params[1]) - self._cut_trap_depth
                if f(max_T) < 0.5: vel_c = brentq(f, 0, max_T, full_output = False)
                else: vel_c = -1
            #--

        return T_c, fit_params

    # Get 2D-histogram of positions removing an axis
    def pos_2Dhist(self, axis = 0, val = 0):
        #
        # Get bin index
        bin_idx = 0
        for idx, bin_size in enumerate(self.pos_3Dhist["bins"][axis]):
            if idx > 0 and (float(val) <= bin_size):
                bin_idx = idx - 1
                break

        #
        # Get bins
        if axis == 0:
            axis_label = {'y', 'z'}
            hist = np.zeros((len(self.pos_3Dhist["bins"][1]), len(self.pos_3Dhist["bins"][2])))

        elif axis == 1:
            axis_label = {'x', 'z'}
            hist = np.zeros((len(self.pos_3Dhist["bins"][0]), len(self.pos_3Dhist["bins"][2])))

        elif axis == 2:
            axis_label = {'x', 'y'}
            hist = np.zeros((len(self.pos_3Dhist["bins"][0]), len(self.pos_3Dhist["bins"][1])))

        #
        # Get densities
        for i in range(len(hist)):
            for j in range(len(hist[i])):
                if axis == 0:
                    hist[i][j] = self.pos_3Dhist["dens"][bin_idx,i,j]

                elif axis == 1:
                    hist[i][j] = self.pos_3Dhist["dens"][i,bin_idx,j]

                elif axis == 2:
                    hist[i][j] = self.pos_3Dhist["dens"][i,j,bin_idx]

        return hist
        path = self.directory + "log.csv"
        pd.Series(data, name="log").to_csv(path, header="log")
