from tetragono.shell import *
from boson_metal.hopping_hamiltonian import hopping_hamiltonians
from boson_metal.restrict_Sz import restrict
from boson_metal.initial_state import initial_configuration

seed(2333)

gm_create("boson_metal", L1=10, L2=10, D=4, J=-1, K=3, mu=-1.6)

gm_conf_create(initial_configuration)

gm_run(
    1000,
    100,
    0.01,
    configuration_cut_dimension=12,
    sampling_method='sweep',
    sweep_hopping_hamiltonians=hopping_hamiltonians,
    restrict_subspace=restrict,
    use_natural_gradient=True,
    momentum_parameter=0.9,
    use_fix_relative_step_size=True,
    log_file="run.log",
)

# gm_conf_dump("conf.dat")
