import configparser
import numpy as np
import torch.optim as optim
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch
import os
import wandb
import PINNs
import shutil
import sys

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)
if len( sys.argv ) < 2:
    raise Exception("Please provid a config file")

#with open(sys.argv[1], 'r') as config_file:
config = configparser.ConfigParser()
config.read(sys.argv[1])
print(sys.argv[1])







torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if len( sys.argv ) > 1:
    model_index = int(sys.argv[2])
    if model_index == 0:
        model_type = "PINN"
    elif model_index == 1:
        model_type = "Global_NSources_Conditioned_Pinns"
    elif model_index == 2:
        model_type = "Relative_Distance_NSources_Conditioned_Pinns"
    elif model_index == 3:
        model_type = "Relative_Distance_FullDomain_Conditioned_Pinns"
    elif model_index == 4:
        model_type = "Global_NSources_Conditioned_Lame_Pinns"
    elif model_index == 5:
        model_type = "Global_FullDomain_Conditioned_Pinns"
    elif model_index == 6:
        model_type = "Global_FullDomain_Conditioned_Pinns_Scramble_Resample"
    else:
        raise Exception("Unsupported model index {} please use a number between 0 and 6 : \n 0 -- PINNs \n 1 -- Global_NSources_Conditioned_Pinns \n 2 -- Relative_Distance_NSources_Conditioned_Pinns \n 3 -- Relative_Distance_FullDomain_Conditioned_Pinns \n 4 -- Global_NSources_Conditioned_Lame_Pinns \n 5 -- Global_FullDomain_Conditioned_Pinns \n -- Global_FullDomain_Conditioned_Pinns_Scramble_Resample ".format(model_index))
else:
    model_type = config['Network']['model_type']

name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
    model_type,
    config['description']['tag'],
    config['Network']['n_points'],
    config['optimizer']['n_epochs'],
    config['optimizer']['lr'],
    config['optimizer']['max_iter'],
    config['optimizer']['max_eval'],
    config['optimizer']['history_size'],
    config['Network']['n_hidden_layers'],
    config['Network']['n_neurons'],
    config['Network']['activation'],
    config['initial_condition']['t1']
)

dir=name
dir_path = os.path.join(os.getcwd(), dir)
print(dir_path)

# Check if the directory already exists
if os.path.exists(dir_path):
    # Remove the existing directory and all its subdirectories
    shutil.rmtree(dir_path)

# Create the directory
os.mkdir(dir_path)
shutil.copyfile("config.ini", dir_path+"/config.ini")

wandb.init(project='Semester Thesis',name = name)

config_for_run = configparser.ConfigParser()
config_for_run.read(dir_path+"/config.ini")




if len( sys.argv ) > 1:
    model_index = int(sys.argv[2])
    if model_index == 0:
        pinn = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as PINNs")
        # Default behaviour to avoid unexpected behaviour
        if config['Network']['conditioning'] == 'True':
            raise Exception("Chose and unconditioned model with conditioning set to True")

        if config['initial_condition']['source_function'] == 'explosion_conditioned':
            raise Exception("Chose and unconditioned model with a condtioned source")
        config_for_run.set('Network', 'model_type', 'Pinns')

    elif model_index == 1:
        pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as Global_NSources_Conditioned_Pinns")
        if config['Network']['conditioning'] == 'False':
            raise Exception("Chose a conditioned model with conditioning set to False")
        config_for_run.set('Network', 'model_type', 'Global_NSources_Conditioned_Pinns')
    elif model_index == 2:
        pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as Relative_Distance_NSources_Conditioned_Pinns")
        config_for_run.set('Network', 'model_type', 'Relative_Distance_NSources_Conditioned_Pinns')
    elif model_index == 3:
        pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as Relative_Distance_FullDomain_Conditioned_Pinns")
        config_for_run.set('Network', 'model_type', 'Relative_Distance_FullDomain_Conditioned_Pinns')
    elif model_index == 4:
        pinn = PINNs.Global_NSources_Conditioned_Lame_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as Global_NSources_Conditioned_Lame_Pinns")
        config_for_run.set('Network', 'model_type', 'Global_NSources_Conditioned_Lame_Pinns')

    elif model_index == 5:
        pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as Global_FullDomain_Conditioned_Pinns")
        config_for_run.set('Network', 'model_type', 'Global_FullDomain_Conditioned_Pinns')

    elif model_index == 6:
        pinn = PINNs.Global_FullDomain_Conditioned_Pinns_Scramble_Resample(int(config['Network']['n_points']), wandb_on=True,config=config)
        print("model type selected as Global_FullDomain_Conditioned_Pinns_Scramble_Resample")
        config_for_run.set('Network', 'model_type', 'Global_FullDomain_Conditioned_Pinns_Scramble_Resample')
    else:
        raise Exception("Unsupported model index {} please use a number between 0 and 6 : \n 0 -- PINNs \n 1 -- Global_NSources_Conditioned_Pinns \n 2 -- Relative_Distance_NSources_Conditioned_Pinns \n 3 -- Relative_Distance_FullDomain_Conditioned_Pinns \n 4 -- Global_NSources_Conditioned_Lame_Pinns \n 5 -- Global_FullDomain_Conditioned_Pinns \n -- Global_FullDomain_Conditioned_Pinns_Scramble_Resample ".format(model_index))

else:
    print("Model type not specified, reading from config file instead")
    model_type = config['Network']['model_type']
    if model_type == "Pinns":
        pinn = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
    elif model_type == "Global_NSources_Conditioned_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
    elif model_type == "Relative_Distance_NSources_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
    elif model_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
    elif model_type == "Global_NSources_Conditioned_Lame_Pinns":
        pinn = PINNs.Global_NSources_Conditioned_Lame_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
    elif model_type == "Global_FullDomain_Conditioned_Pinns":
        pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
    else:
        raise Exception("Model type {} not supported".format(model_type))

#pinn = PINNs.Pinns_with_helper(int(config['Network']['n_points']), wandb_on=True,config=config)

with open(dir_path+"/config.ini", 'w') as configfile2:
    config_for_run.write(configfile2)
print({section: dict(config[section]) for section in config.sections()})
n_epochs = int(config['optimizer']['n_epochs'])
#optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),lr=1e-3)
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                                lr=float(float(config['optimizer']['lr'])),
                                max_iter=int(config['optimizer']['max_iter']),
                                max_eval=int(config['optimizer']['max_eval']),
                                history_size=int(config['optimizer']['history_size']),
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-8,
                                tolerance_change=1.0 * np.finfo(float).eps)

#pinn.approximate_solution.load_state_dict(torch.load('FLIPY_NEW_PINN_LBFGS_no_init_120000_70_1.0_200_200_800_3_64_tanh_0.07.pth', map_location=torch.device('cpu')))

#helper_network = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=True,config=config)
#helper_network.approximate_solution.load_state_dict(torch.load('FLIPY_NEW_PINN_LBFGS_no_init_120000_70_1.0_200_200_800_3_64_tanh_0.07.pth', map_location=torch.device('cpu')))
hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)



torch.save(pinn.approximate_solution.state_dict(),'{}/model.pth'.format(name))

