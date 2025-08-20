# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add LatentDynamics, Physics directories to the search path.
import  sys;
import  os;
LD_Path         : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "LatentDynamics"));
Physics_Path    : str = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
sys.path.append(LD_Path); 
sys.path.append(Physics_Path); 

import  logging;

import  numpy;
import  torch; 

import  Burgers;
import  Explicit;
from    LatentDynamics      import  LatentDynamics;
from    SINDy               import  SINDy;
from    ParameterSpace      import  ParameterSpace;
from    GPLaSDI             import  BayesianGLaSDI;
from    Model               import  Autoencoder, load_Autoencoder;
from    Physics             import  Physics;
from    Advection           import  Advection;
from    Burgers2D           import  Burgers2D;

# Set up logger.
LOGGER  : logging.Logger    = logging.getLogger(__name__);

# Set up the dictionaries; we use this to allow the code to call different classes, functions 
# depending on the settings.
trainer_dict    =  {'gplasdi'               : BayesianGLaSDI};
model_dict      =  {'ae'                    : Autoencoder,
                    'autoencoder'           : Autoencoder};
model_load_dict =  {'ae'                    : load_Autoencoder,
                    'autoencoder'           : load_Autoencoder};
ld_dict         =  {'sindy'                 : SINDy};
physics_dict    =  {'Burgers'               : Burgers.Burgers,
                    'Burgers2D'             : Burgers2D,
                    'Explicit'              : Explicit.Explicit,
                    'Advection'             : Advection};



# -------------------------------------------------------------------------------------------------
# Initialization functions
# -------------------------------------------------------------------------------------------------

def Initialize_Trainer(config : dict, restart_dict : dict = {}) -> tuple[BayesianGLaSDI, ParameterSpace, Physics, torch.nn.Module, LatentDynamics]:
    """
    Initialize a LaSDI object with a latent space model and physics object according to config 
    file. Currently only 'gplasdi' is available.

    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config: dict
        The dictionary that we loaded from a .yml file. It should house all the settings we expect 
        to use to generate the data and train the models. We expect this dictionary to contain the 
        following keys (if a key is within a dictionary that is specified by another key, then we 
        tab the sub-key relative to the dictionary key): 
            - physics           (used by "initialize_physics")
                - type
            - latent_dynamics   (how we parameterize the latent dynamics; e.g. SINDy)
                - type
            - lasdi
                - type

    restart_dict : dict, optional
        If provided, then we will use the settings in this dictionary to initialize the trainer, 
        parameter space, physics, model, and latent dynamics. If not provided, then we will 
        initialize everything from scratch.
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    trainer, param_space, physics, model, latent_dynamics
     
    trainer : BayesianGLaSDI
        Should have been initialized using the settings in config and is ready to begin training.

    param_space : ParameterSpace
        holds the combinations of parameters in the testing and training sets.
     
    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters.

    model : torch.nn.Module
        The model we use to map between the FOM and ROM spaces. Specifically, the model can 
        encode a snapshot/frame (measurement at a specific time) of the FOM solution to its 
        corresponding ROM frame. It can also decode a ROM frame back to a FOM frame. The n_IC 
        attribute of this object must match that of latent_dynamics.

    latent_dynamics : LatentDynamics
        Defines the dynamical system in model's latent space. The n_IC attribute of this object 
        must match the n_IC attribute of model.
    """

    # Fetch the trainer type. Note that only "gplasdi" is allowed.
    trainer_type            = config['lasdi']['type'];
    assert(trainer_type in config['lasdi']);
    assert(trainer_type in trainer_dict);
    LOGGER.info("Initializing Trainer (%s)" % trainer_type);

    # Set up a ParameterSpace object. This will keep track of all parameter combinations we want
    # to try during testing and training. We load the set of possible parameters and their possible
    # values using the configuration file. If we are using a restart file, then load it's 
    # ParameterSpace object.
    param_space = ParameterSpace(config);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        param_space.load(restart_dict['parameter_space']);
    
    # Get the "physics" object we use to generate the FOM dataset.
    physics : Physics   = Initialize_Physics(config, param_space.param_names);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        physics.load(restart_dict['physics']);

    # Get the model (autoencoder). We try to learn dynamics that describe how the latent space of
    # this model evolve over time. If we are using a restart file, then load the saved model 
    # parameters from file.
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        model_type : str    = config['model']['type'];
        model               = model_load_dict[model_type](restart_dict['model']);
    else: 
        model               = Initialize_Model(physics, config);

    # Initialize the latent dynamics model. If we are using a restart file, then load the saved
    # latent dynamics from this file. 
    ld_type                 = config['latent_dynamics']['type'];
    assert(ld_type in config['latent_dynamics']);
    assert(ld_type in ld_dict);
    latent_dynamics         = ld_dict[ld_type]( n_z             = model.n_z, 
                                                coef_norm_order = config['latent_dynamics']['coef_norm_order'],
                                                Uniform_t_Grid  = physics.Uniform_t_Grid);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        latent_dynamics.load(restart_dict['latent_dynamics']);

    # Initialize the trainer object. If we are using a restart file, then load the 
    # trainer from that file.
    trainer                 = trainer_dict[trainer_type](physics, model, latent_dynamics, param_space, config['lasdi'][trainer_type]);
    if (bool(restart_dict) == True):        # Empty dictionaries evaluate to False. restart_dict is empty if we are not using a restart file.
        trainer.load(restart_dict['trainer']);

    # If we are loading from a restart file, then make a checkpoint using the current model 
    # parameters. This enables us to run get_new_sample_point, which loads from a checkpoint.       
    if (bool(restart_dict) == True): 
        torch.save(model.cpu().state_dict(), trainer.path_checkpoint + '/' + 'checkpoint.pt');


    # All done!
    return trainer, param_space, physics, model, latent_dynamics;



def Initialize_Model(physics : Physics, config : dict) -> torch.nn.Module:
    """
    Initialize a model (autoencoder) according to config file. 
    

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters. 

    config : dict
        This should be a dictionary that we loaded from a .yml file. It should house all the 
        settings we expect to use to generate the data and train the models. We expect this 
        dictionary to contain the following keys (if a key is within a dictionary that is specified 
        by another key, then we tab the sub-key relative to the dictionary key): 
            - model
                - type
    
       
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    model : torch.nn.Module
        A torch.nn.Module object that acts as the trainable model in the gplasdi framework. This 
        model should have a latent space of some form. We learn a set of dynamics to describe how 
        this latent space evolves over time. 
    """


    # First, determine what model we are using in the latent dynamics. Make sure the user 
    # included all the information that is necessary to initialize the corresponding dynamics.
    model_type : str = config['model']['type'];
    assert(model_type in config['model']);
    assert(model_type in model_dict);
    LOGGER.info("Initializing Model (%s)" % model_type);

    # Autoencoder case
    if(model_type == "ae" or model_type == "autoencoder"):
        # Next, fetch the hidden widths and latent dimension (n_z). 
        model_config        : dict              = config['model'][model_type];
        hidden_widths       : list[int]         = model_config['hidden_widths'];
        n_z                 : int               = model_config['latent_dimension'];

        # Fetch the activations. This can either be a string or a list of strings. If it's 
        # a string, then we use that activation for all layers.
        n_hidden_layers     : int               = len(hidden_widths);
        if(isinstance(model_config['activations'], str)):
            activations         : list[str]     = [model_config['activations']] * n_hidden_layers;   # The final layer has no activation.
        elif(isinstance(model_config['activations'], list)):
            activations         : list[str]     = model_config['activations'];
            assert(len(activations) == n_hidden_layers);
        else:
            raise ValueError("Activations must be a string or a list of strings.");

        # Now build the widths attribute + fetch Frame_Shape from physics.
        Frame_Shape         : list[int]         = physics.Frame_Shape;
        space_dim           : int               = numpy.prod(Frame_Shape).item();
        widths              : list[int]         = [space_dim] + hidden_widths + [n_z];

        # Now build the model!
        model           : torch.nn.Module       = model_dict[model_type](
                                                        widths          = widths, 
                                                        activations     = activations, 
                                                        reshape_shape   = Frame_Shape);

        # All done!
        return model;

    else:
        raise ValueError("Model type %s not supported." % model_type);



def Initialize_Physics(config: dict, param_names : list[str]) -> Physics:
    '''
    Initialize a physics FOM model according to config file.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    config : dict
        A dictionary we loaded from a .yml file. It should house all the settings we expect to use 
        to generate the data and train the models. We expect this dictionary to contain the 
        following keys (if a key is within a dictionary that is specified by another key, then we 
        tab the sub-key relative to the dictionary key): 
            - physics 
                - type

    param_names : list[str], len  = n_p
        A list housing the names of the parameters in the physics model. There should be an entry 
        in the configuration file for each named parameter. 
            
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    physics : Physics
        Encodes the FOM model. It allows us to fetch the FOM solution and/or initial conditions 
        for a particular combination of parameters. Initialized using the n_p parameters in the 
        config['physics'] dictionary. 
    '''

    # First, determine what kind of "physics" object we want to load.
    physics_cfg     : dict      = config['physics'];
    physics_type    : str       = physics_cfg['type'];
    LOGGER.info("Initializing Physics (%s)" % physics_type);

    # Next, initialize the "physics" object we are using to build the simulations.
    physics         : Physics   = physics_dict[physics_type](physics_cfg, param_names);

    # All done!
    return physics;
