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
tet.common_toolkit.write_to_file(su_lattice, "/dev/null")

su_lattice.update(100, 0.01, 5)

# showln is a helper function, which will only call print in proc 0
ex_lattice = tet.conversion.simple_update_lattice_to_exact_state(su_lattice)
tet.common_toolkit.showln("Exact energy is", ex_lattice.observe_energy())
ex_lattice.update(100, 4)
tet.common_toolkit.showln("Exact energy is", ex_lattice.observe_energy())

gm_lattice = tet.conversion.simple_update_lattice_to_sampling_lattice(su_lattice)
# To run gradient, create observer first
observer1 = tet.Observer(gm_lattice,
                         enable_energy=True,
                         enable_gradient=True,
                         enable_natural_gradient=True,
                         observer_set={"Sz": kitaev.Sz.measurement(gm_lattice)})
# You can create another observer
observer2 = tet.Observer(gm_lattice, enable_energy=True, enable_gradient=True, enable_natural_gradient=True)
# Run gradient
for grad_step in range(10):
    # Choose observer
    if grad_step % 2 == 0:
        observer = observer1
    else:
        observer = observer2
    # Prepare sampling environment
    with tet.common_toolkit.seed_differ, observer:
        # create sampling object and do sampling
        sampling = tet.DirectSampling(gm_lattice, cut_dimension=8, restrict_subspace=None, double_layer_cut_dimension=4)
        for sampling_step in range(1000):
            observer(*sampling())
    tet.common_toolkit.showln("grad", grad_step, *observer.energy)
    # Get Sz measure result
    if observer == observer1:
        tet.common_toolkit.showln("   Sz:", observer.result["Sz"])
    # Get gradient
    grad = observer.natural_gradient(step=20, epsilon=0.01)
    # Maybe you want to use momentum
    if grad_step == 0:
        total_grad = grad
    else:
        total_grad = total_grad * 0.9 + grad * 0.1
    # Fix relative step size
    param = observer.fix_relative_parameter(total_grad)
    # Apply gradient
    gm_lattice._lattice -= 0.01 * param * tet.common_toolkit.lattice_conjugate(total_grad)
    # Fix gauge
    gm_lattice.expand_dimension(1.0, 0)
    # Bcast buffer to avoid numeric error
    tet.common_toolkit.bcast_lattice_buffer(gm_lattice._lattice)
    # Maybe you want to save file
    tet.common_toolkit.write_to_file(gm_lattice, "/dev/null")

# low level api usage TODO
# + easy usage of sweep
# + line search
# + momentum orthogonalize
# + simplify gradient with tetragono low level api
