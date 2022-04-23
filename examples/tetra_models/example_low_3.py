import TAT
import tetragono as tet
from boson_metal import create

TAT.random.seed(2333)

# Create abstrace lattice first and cast it to su lattice
abstract_lattice = create(L1=10, L2=10, D=4, J=-1, K=3, mu=-1.6)
gm_lattice = tet.SamplingLattice(abstract_lattice)

# To run gradient, create observer first
observer = tet.Observer(
    gm_lattice,
    enable_energy=True,
    enable_gradient=True,
    enable_natural_gradient=True,
)


def classical_energy(configuration):
    n_up = 0
    for l1 in range(configuration._owner.L1):
        for l2 in range(configuration._owner.L2):
            if configuration[l1, l2, 0][1] == 0:
                n_up += 1
    return (n_up - 37)**2


observer.set_classical_energy(classical_energy)
for grad_step in range(100):
    # Prepare sampling environment
    with tet.seed_differ, observer:
        # create sampling object and do sampling
        sampling = tet.DirectSampling(gm_lattice,
                                      cut_dimension=12,
                                      restrict_subspace=None,
                                      double_layer_cut_dimension=4)
        for sampling_step in range(1000):
            if sampling_step % tet.mpi_size == tet.mpi_rank:
                possibility, configuration = sampling()
                observer(possibility, configuration)
                tet.show("sampling", sampling_step, "current energy is", *observer.energy)
    tet.showln("grad", grad_step, *observer.energy)
    # Get gradient
    grad = observer.natural_gradient(step=20, epsilon=0.01)
    # Maybe you want to use momentum
    if grad_step == 0:
        total_grad = grad
    else:
        total_grad = total_grad * 0.9 + grad * 0.1
    # Apply gradient
    gm_lattice._lattice -= 0.01 * gm_lattice.fix_relative_to_lattice(total_grad)
    # Normalzie state
    observer.normalize_lattice()
