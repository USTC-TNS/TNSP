import pickle
import TAT
import tetragono as tet
import kitaev
import kitaev.Sz

TAT.random.seed(2333)

# Create abstrace lattice first and cast it to su lattice
abstract_lattice = kitaev.create(L1=2, L2=2, D=4, Jx=1, Jy=1, Jz=1)
su_lattice = tet.SimpleUpdateLattice(abstract_lattice)

# Use pythonic style, aka, piclle, to save data.
# But should check mpi rank when saving data.
if tet.common_variable.mpi_rank == 0:
    with open("/dev/null", "wb") as file:
        pickle.dump(su_lattice, file)

su_lattice.update(100, 0.01, 5)

# showln is a helper function, which will only call print in proc 0
ex_lattice = tet.conversion.simple_update_lattice_to_exact_state(su_lattice)
tet.common_variable.showln("Exact energy is", ex_lattice.observe_energy())
ex_lattice.update(100, 4)
tet.common_variable.showln("Exact energy is", ex_lattice.observe_energy())

gm_lattice = tet.conversion.simple_update_lattice_to_sampling_lattice(su_lattice)
# To run gradient, create observer first
observer = tet.Observer(gm_lattice, restrict_subspace=None)
observer.add_energy()
observer.enable_gradient()
observer.enable_natural_gradient()
# The measurement name is customed in fact
observer.add_observer("Sz", kitaev.Sz.measurement(gm_lattice))
# Run gradient
for grad_step in range(10):
    # Prepare sampling environment
    with tet.common_variable.seed_differ, observer:
        # create sampling object and do sampling
        sampling = tet.DirectSampling(gm_lattice, cut_dimension=8, restrict_subspace=None, double_layer_cut_dimension=4)
        for sampling_step in range(100):
            observer(*sampling())
    tet.common_variable.showln("grad", grad_step, *observer.energy)
    # Get Sz measure result
    tet.common_variable.showln("   Sz:", observer.result["Sz"])
    # Get gradient
    grad = observer.natural_gradient(step=20, epsilon=0.01)
    # Maybe you want to use momentum
    if grad_step == 0:
        total_grad = grad
    else:
        for l1 in range(gm_lattice.L1):
            for l2 in range(gm_lattice.L2):
                total_grad[l1][l2] = total_grad[l1][l2] * 0.9 + grad[l1][l2] * 0.1
    # Fix relative step size
    param = observer.fix_relative_parameter(total_grad)
    # Apply gradient
    for l1 in range(gm_lattice.L1):
        for l2 in range(gm_lattice.L2):
            gm_lattice[l1, l2] -= 0.01 * param * total_grad[l1][l2].conjugate(positive_contract=True)
    # Maybe you want to save file
    if tet.common_variable.mpi_rank == 0:
        with open("/dev/null", "wb") as file:
            pickle.dump(gm_lattice, file)

# low level api usage TODO
# + easy usage of sweep
# + line search
# + momentum orthogonalize
# + fix gauge
# + normalize
# + all in shell_commands package
