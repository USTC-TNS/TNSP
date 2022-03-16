from tetragono.shell import *

# set random seed
seed(2333)

# what you can create is lattice for simple update only
# different model have different parameter for create
su_create("kitaev", L1=2, L2=2, D=4, Jx=1, Jy=1, Jz=1)

# save or open file
su_dump("/dev/null")
# su_load xxx

# total_step, step_size, new_dimension
su_update(400, 0.01, 5)
# for system size > 4*4, it is dangerous to get exact state
su_to_ex()
ex_energy()
ex_update(1000, 4)
ex_energy()

su_to_gm()
gm_run(100,
       10,
       0.01,
       configuration_cut_dimension=8,
       use_natural_gradient=True,
       use_line_search=True,
       log_file="run.log",
       measurement="kitaev.Sz")
