import numpy as np
import torch
import random
import json 

# HOME_DIR = "/scratch/thaihung/project_data/Grant25/material_data"
HOME_DIR = "/scratch/s223540177/project_data/Grant25/material_data"


# mat_name = "VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_1000K.OUTCAR"

# ['VASP_LGPS_mp-696128_conventional_standard_MD_repeat_600K.OUTCAR.json', 
# 'VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_800K.OUTCAR.json', 
# 'VASP_LGPS_mp-696128_conventional_standard_MD_repeat_1200K.OUTCAR.json', 
# 'VASP_LGPS_mp-696128_conventional_standard_MD_repeat_1000K.OUTCAR.json', 
# 'VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_600K.OUTCAR.json', 
# 'VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_1200K.OUTCAR.json', 
# 'VASP_LGPS_mp-696128_conventional_standard_MD_repeat_800K.OUTCAR.json', 
# 'VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_1000K.OUTCAR.json']

# energy_threshold_pair_old = {
#     "Ge_Ge": -3.81983811458227,
#     "Li_S": -3.583609574976967,
#     "S_S": -5.781557916705929,
#     "Li_Ge": -2.254359970947006,
#     "P_Ge": -5.166113958829331,
#     "P_S": -6.3350552362801755,
#     "Li_P": -3.167438922197101,
#     "Li_Li": -1.0554370708554515,
#     "P_P": -7.055604569657446,
#     "S_Ge": -6.378262017713406,
#     "Li_Ta": -4.186531201821578,
#     "Se_I": -2.942401591871086,
#     "O_Cs": -4.211743706523936,
#     "O_Ta": -7.99999769314658,
#     "Li_I": -3.4033679888902393,
#     "Se_In": -3.6097683778991376,
#     "Li_Cs": -0.6495083160269904,
#     "I_Ta": -5.890631085146204,
#     "Li_Zn": -0.49943641597505994,
#     "Zn_Zn": -0.2130896732085693,
#     "Li_In": -1.1724076616001682,
#     "Ta_Ta": -7.999970571372166,
#     "Li_Br": -3.9313036350281454,
#     "Br_In": -3.7298870923256175,
#     "Br_Br": -2.2555566310443904,
#     "O_O": -7.893892737784694,
#     "O_Se": -6.403988752833714,
#     "O_In": -5.105897009233641,
#     "F_Ta": -7.999014005036881,
#     "Zn_Ta": -3.3458403404408354,
#     "Li_F": -5.7518218684658144,
#     "Br_I": -2.1872620796371005,
#     "Ga_I": -3.5875196707186365,
#     "Zn_Cs": -1.788178515789483,
#     "Li_O": -4.798863275390404,
#     "F_Se": -4.106217246411156,
#     "O_I": -4.230391826756075,
#     "Se_Ta": -7.999928589891302,
#     "Ga_Ta": -4.946208884555311,
#     "Zn_Br": -1.6733417702194453,
#     "Zn_In": -0.5773564354591075,
#     "O_F": -4.099826596523718,
#     "Br_Ta": -6.354241180273405,
#     "F_F": -2.2496689799290643,
#     "O_Zn": -3.0362788237505662,
#     "Li_Ga": -1.1807974260818168,
#     "In_I": -3.4509025367441435,
#     "Se_Se": -4.677895280157521,
#     "I_I": -2.049066234960244}


energy_threshold_pair = {
    "Ge_Ge": -1.7099695205688477, # updated
    "Li_S": -1.663771152496338, # updated
    "S_S": -2.0867786407470703,
    "Li_Ge": -1.2451030015945435,  # updated
    "P_Ge": -2.593142032623291, # updated
    "P_S": -2.580552577972412,
    "Li_P": -2.027696132659912, # updated
    "Li_Li": -0.1017923355102539,            # updated
    "P_P": -3.1617074012756348, # updated
    "S_Ge": -2.0267767906188965, # updated
    "Li_Ta": -3.309340000152588,              # updated
    "Se_I": -2.942401591871086,
    "O_Cs": -4.211743706523936,
    "O_Ta": -7.99999769314658,                # not updated
    "Li_I": -3.4033679888902393,
    "Se_In": -3.6097683778991376,
    "Li_Cs": -0.6495083160269904,
    "I_Ta": -5.890631085146204,
    "Li_Zn": -0.49943641597505994,
    "Zn_Zn": -0.2130896732085693,
    "Li_In": -1.1724076616001682,
    "Ta_Ta": -7.999970571372166,              # not updated
    "Li_Br": -1.8012025356292725,
    "Br_In": -3.7298870923256175,
    "Br_Br": -0.6903369426727295,
    "O_O": -3.866206169128418,                # updated
    "O_Se": -6.403988752833714,
    "O_In": -5.105897009233641,
    "F_Ta": -7.999014005036881,
    "Zn_Ta": -3.3458403404408354,
    "Li_F": -5.7518218684658144,
    "Br_I": -2.1872620796371005,
    "Ga_I": -3.5875196707186365,
    "Zn_Cs": -1.788178515789483,
    "Li_O": -2.9619300365448,                 # updated
    "F_Se": -4.106217246411156,
    "O_I": -4.230391826756075,
    "Se_Ta": -7.999928589891302,
    "Ga_Ta": -4.946208884555311,
    "Zn_Br": -1.6733417702194453,
    "Zn_In": -0.5773564354591075,
    "O_F": -4.099826596523718,
    "Br_Ta": -6.354241180273405,
    "F_F": -2.2496689799290643,
    "O_Zn": -3.0362788237505662,
    "Li_Ga": -0.6013139486312866,
    "In_I": -3.4509025367441435,
    "Se_Se": -4.677895280157521,
    "I_I": -2.049066234960244,
    "Br_Ga": -2.75104022026062,
    "Br_Li": -1.8012025356292725,
    "Ga_Li": -0.6013139486312866,
    "Ga_Br": -2.75104022026062,
    "Ga_Ga": -0.827075719833374, # added
    "Ge_P": -2.593142032623291, # added
    "Ge_Li": -1.2451030015945435, # added
    "S_Li": -1.663771152496338, # added
    "P_Li": -2.027696132659912, # added
    "S_P": -2.580552577972412, # added
    "Ge_S": -2.0267767906188965, # added
}

#### Max energy threshold for each pair find from data
#### Or * with a threshold coefficient.

def load_energy_files(mat_name):
    # atom_coef={
    # "Ge_Ge": [ 3.06553609,  2.31716164,  1.33097709, -4.77488633],
    # "Li_S": [ 2.8157454,   2.1906933,   1.03889333, -4.47951653],
    # "S_S": [ 5.14019036,  1.84741248,  1.89367176, -7.22696855],
    # "Li_Ge": [ 1.57287294,  2.4178624,   1.09224158, -2.81797605],
    # "P_Ge": [ 3.86452153,  2.08948594,  1.56556993, -6.45766159],
    # "P_S": [ 5.33971788,  1.87961288,  1.79226186, -7.9202706 ],
    # "Li_P": [ 1.93163744,  2.41853734,  0.96305791, -3.95933342],
    # "Li_Li": [ 0.85718379,  2.62028277,  0.93189134, -1.31930903],
    # "P_P": [ 5.65857861,  1.90210783,  1.69104392, -8.82028552],
    # "S_Ge": [ 5.94695179,  1.99753768,  1.60163267, -7.97372863]
    # }

    with open(f'{HOME_DIR}/md/{mat_name}.atom_coef.json') as f:
        atom_coef = json.load(f)

    with open(f'{HOME_DIR}/md/{mat_name}.atoms.json') as f:
        atom_list = json.load(f)
        print(atom_list)
        atom2index = {}
        atom_set = list(set(atom_list))

        for i in range(len(atom_set)):
            for j in range(len(atom_set)):
                if f"{atom_set[i]}_{atom_set[j]}" not in atom_coef:
                    atom_coef[f"{atom_set[i]}_{atom_set[j]}"]=atom_coef[f"{atom_set[j]}_{atom_set[i]}"]
        print(atom_coef)

        for ia, a in enumerate(atom_list):
            if a in atom2index:
                atom2index[a].append(ia)
            else:
                atom2index[a] = [ia]

    print(atom_set)
    print(atom2index)
    return atom_set, atom2index, atom_coef

def tensor2pos(row):
    """Calculate the starting position of the next timestep t given the position at timestep t-1 and displacement t-1"""
    # eg: (16, 300) -> (16,150)
    dim = row.shape[-1]
    pos = row[:, :dim//2] + row[:, dim//2:]
    return pos

def cal_pos(x, y):
    # Calculate the culmulative sum displacement accross the predicted sequence y (dim=1). 
    # For example: 32 timesteps will sum over 32. Each timestep will have a value of cumulative displacement associated.
    # Eg: (16,32,150) -> (16,32,150)
    ycum = torch.cumsum(y, dim=1)
    # The expand is used to repeat the tensor thus giving same init position for 16 items in batch
    pos_start = tensor2pos(x[:,-1,:]).unsqueeze(1).expand(-1, y.shape[1], -1)
    pos = pos_start + ycum
    return pos.view(pos.shape[0], pos.shape[1], pos.shape[2]//3, 3)

def sample_atom_pair_index(atom_set, atom2index):
    a1 = random.choice(atom_set)
    a2 = random.choice(atom_set)
    i1 = random.choice(atom2index[a1])
    i2 = random.choice(atom2index[a2])
    if a1==a2:
        while i2==i1:
            i2 = random.choice(atom2index[a2])
    return i1, i2, a1, a2   


def cal_energy_loss(x, y, atom_set, atom2index, atom_coef, num_samples=100, debug=False):
    """"This is the Physics-Informed Loss Function
        Currently, this function samples 100 points from x and y to calculate the loss
    """
    for i in range(num_samples):
        i1, i2, a1, a2 = sample_atom_pair_index(atom_set, atom2index)
        if debug: 
            print("index atom 1", i1)
            print("index atom 2", i2)
            print("atom 1", a1)
            print("atom 2", a2)
            
        pos = cal_pos(x, y)
        if debug:
            print("x shape", x.shape)
            print("y shape", y.shape)
            print("final_position_shape", pos.shape)
            print("pos atom 1", pos[:,:,i1,:].shape)
            print("pos atom 2", pos[:,:,i2,:].shape)
            # exit()

        dis = torch.sum((pos[:,:,i1,:]-pos[:,:,i2,:])**2,dim=-1)**0.5     # Is this distance correct or should we take the max between each axis?      
        if i==0:
            e =  energy(atom_coef, a1, a2, dis)
        else:
            e += energy(atom_coef, a1, a2, dis)
    return e/num_samples 

def cal_energy_loss_only_for_energy_greater_than_threshold(x, y, atom_set, atom2index, atom_coef, num_samples=100, debug=False):
    """"This is the Physics-Informed Loss Function
        Currently, this function samples 100 points from x and y to calculate the loss
    """
    e = None
    for i in range(num_samples):
        i1, i2, a1, a2 = sample_atom_pair_index(atom_set, atom2index)
        if debug: 
            print("index atom 1", i1)
            print("index atom 2", i2)
            print("atom 1", a1)
            print("atom 2", a2)
            
        pos = cal_pos(x, y)
        if debug:
            # Why the 3rd dimension is 50 infead of 49? (OK the atom starts from 0)
            print("x shape", x.shape)
            print("y shape", y.shape)
            print("final_position_shape", pos.shape)
            print("pos atom 1", pos[:,:,i1,:].shape)
            print("pos atom 2", pos[:,:,i2,:].shape)
            # exit()

        # Wouldnt this formula calculate the displacement over all timestep predicted? Is it the same with calculating only for the last timestep?
        # Maybe not? 
        dis = torch.sum((pos[:,:,i1,:]-pos[:,:,i2,:])**2,dim=-1)**0.5     # Is this distance correct or should we take the max between each axis?  
        e_val = energy(atom_coef, a1, a2, dis)  # Compute once
        pair = f"{a1}_{a2}" if energy_threshold_pair.get(f"{a1}_{a2}") else f"{a2}_{a1}"

        try:
            if e_val.max() > energy_threshold_pair.get(pair):
                if e is None:
                    e = e_val
                else:
                    e += e_val
        except:
            print("index atom 1", i1)
            print("index atom 2", i2)
            print("atom 1", a1)
            print("atom 2", a2)
            
    if e is None:
        return torch.tensor(0)
    return (e/num_samples).mean()

def energy_check_during_infer(x, y, atom_set, atom2index, atom_coef, num_samples=100, debug=False):
    """"This is the Physics-Informed Loss Function
        Currently, this function samples 100 points from x and y to calculate the loss
    """
    e = None
    for i in range(num_samples):
        i1, i2, a1, a2 = sample_atom_pair_index(atom_set, atom2index)
        if debug: 
            print("index atom 1", i1)
            print("index atom 2", i2)
            print("atom 1", a1)
            print("atom 2", a2)
            
        pos = cal_pos(x, y)
        if debug:
            # Why the 3rd dimension is 50 infead of 49? (OK the atom starts from 0)
            print("x shape", x.shape)
            print("y shape", y.shape)
            print("final_position_shape", pos.shape)
            print("pos atom 1", pos[:,:,i1,:].shape)
            print("pos atom 2", pos[:,:,i2,:].shape)
            # exit()

        # Wouldnt this formula calculate the displacement over all timestep predicted? Is it the same with calculating only for the last timestep?
        # Maybe not? 
        dis = torch.sum((pos[:,:,i1,:]-pos[:,:,i2,:])**2,dim=-1)**0.5     # Is this distance correct or should we take the max between each axis?  
        e_val = energy(atom_coef, a1, a2, dis)  # Compute once
        pair = f"{a1}_{a2}" if energy_threshold_pair.get(f"{a1}_{a2}") else f"{a2}_{a1}"

        try:
            if e_val.max() > energy_threshold_pair.get(pair):
                # print("index atom 1", i1)
                # print("index atom 2", i2)
                # print("atom 1", a1)
                # print("atom 2", a2)   
                return False
        except:
            print("index atom 1", i1)
            print("index atom 2", i2)
            print("atom 1", a1)
            print("atom 2", a2)   
    return True




def calculate_energy_for_2_atoms_report_energy(timestep, index_timestep, atom_set, atom2index, atom_coef, num_samples=100, debug=False, compare_thres=False):
    """The lower the distance the higher the energy"""
    # print(sampled_timesteps.shape)
    timestep = timestep.reshape(timestep.shape[0], timestep.shape[1] // 3, 3)
    # print(sampled_timesteps[0])
    count = 0
    max_e = 0
    distance = []
    for _ in range(num_samples):
        i1, i2, a1, a2 = sample_atom_pair_index(atom_set, atom2index)

        pos_i1 = torch.tensor(timestep[:, i1, :])  
        pos_i2 = torch.tensor(timestep[:, i2, :])  
        dis = torch.sum((pos_i1-pos_i2)**2,dim=-1)**0.5    
        distance.append(dis)
        eng = energy(atom_coef, a1, a2, dis)        
    
        for i in range(len(eng)):
            pair = f"{a1}_{a2}" if energy_threshold_pair.get(f"{a1}_{a2}") else f"{a2}_{a1}"

            thres = energy_threshold_pair.get(pair) if compare_thres else 0
            if debug:
                print(pair)
                print(thres)
            if eng[i] > thres: 
                count += 1
                if debug:
                    print(f"Index timestep {index_timestep}")
                    print(f"Atom {a1} and Atom {a2}")
                    print(f"Position {i1} and {i2}")
                    print("Distance", dis[i])
                    print("Position", pos_i1[i], pos_i2[i])
                    print("Energy", eng[i])
                if eng[i] > max_e:
                    max_e = eng[i]
    return count, max_e, distance


def calculate_max_thres_energy_in_test_file(timestep, index_timestep, atom_set, atom2index, atom_coef, num_samples=100):
    """The lower the distance the higher the energy"""
    timestep = timestep.reshape(timestep.shape[0], timestep.shape[1] // 3, 3)
    max_energy_dict = {}
    sampled_pairs = set()

    while len(sampled_pairs) < num_samples:
        
        i1, i2, a1, a2 = sample_atom_pair_index(atom_set, atom2index)

        # Sort to avoid both (i1, i2) and (i2, i1) being treated as different
        sample_key = tuple(sorted((i1, i2)))

        # If already sampled, skip
        if sample_key in sampled_pairs:
            continue
        sampled_pairs.add(sample_key)

        if len(sampled_pairs) > 0 and len(sampled_pairs) % 100 == 0:
            print(f"Sampled {len(sampled_pairs)} unique atom pairs...")

        pos_i1 = torch.tensor(timestep[:, i1, :])  
        pos_i2 = torch.tensor(timestep[:, i2, :])  
        dis = torch.sum((pos_i1-pos_i2)**2,dim=-1)**0.5    
        eng = energy(atom_coef, a1, a2, dis) 

        key = f"{a1}_{a2}"
        # Update with max if key exists, else just assign
        if key in max_energy_dict:
            max_energy_dict[key] = torch.maximum(max_energy_dict[key], eng)
        else:
            max_energy_dict[key] = eng
    return max_energy_dict


def energy(atom_coef, atom1, atom2, distance):
    coef = atom_coef[f"{atom1}_{atom2}"]
    m = morse(distance, coef[0], coef[1], coef[2], coef[3]) 
    return m


def morse(x, D_e, x_e, a, b): 
    """x: the interatomic distance.
       x_e: the equilibrium bond length"""
    return D_e* (1 - torch.exp(-a* (x -x_e)))**2 + b
    # return (x -x_e)**2

