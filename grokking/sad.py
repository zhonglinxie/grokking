from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
import copy


def lanczos(matrix_vector, dim: int, neigs: int=6):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module,
                            X: Tensor, y: Tensor, neigs: int=6):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, X, y, delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals, evecs


def compute_hvp(network: nn.Module, loss_fn: nn.Module, X: Tensor, y: Tensor, vector: Tensor):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(y)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    # for (X, y) in iterate_dataset(dataset, physical_batch_size):
    loss = loss_fn(network(X), y) / n
    # print(loss.grad_fn)
    # for param in network.parameters():
    #     print(param.requires_grad)
    grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
    hvp += parameters_to_vector(grads)
    return hvp

def get_hessian_eigenvalues_pow(network: nn.Module, loss_fn: nn.Module,
                                X: Tensor, y: Tensor, num_iterations: int=10,
                                neigs: int=6, tol: float=1e-6):
    v = torch.randn(len(parameters_to_vector(network.parameters())), dtype=torch.float, device='cuda')
    largest_eigenvalue_old = 0.0
    for _ in range(num_iterations):
        with torch.enable_grad():
            Hv = compute_hvp(network, loss_fn, X, y, v)
        v = Hv / torch.norm(Hv)
        largest_eigenvalue = v @ Hv
        # Check for convergence
        if torch.abs(largest_eigenvalue - largest_eigenvalue_old) < tol:
            break

        largest_eigenvalue_old = largest_eigenvalue.clone()
    return largest_eigenvalue, v

class HessianLargestEigenvalue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params_vec, network: nn.Module, loss_fn: nn.Module,
                X: Tensor, y: Tensor, neigs: int=1) -> torch.Tensor:
        # Compute the largest eigenvalue using power iteration method
        # vector_to_parameters(params_vec, network.parameters())

        cloned_network = copy.deepcopy(network)
        cloned_params = params_vec.clone().detach()
        vector_to_parameters(cloned_params, cloned_network.parameters())
        with torch.enable_grad():
            evals, evecs = get_hessian_eigenvalues_pow(cloned_network, loss_fn, X, y, neigs)

        # Save input and eigenvector for backward pass
        ctx.save_for_backward(evals, evecs, parameters_to_vector(network.parameters()))
        ctx.network = network
        ctx.loss_fn = loss_fn
        ctx.X = X
        ctx.y = y

        return evals

    @staticmethod
    def backward(ctx, grad_output):
        evals, evec, parameters = ctx.saved_tensors
        # evec = evecs[:,0]
        network = ctx.network
        loss_fn = ctx.loss_fn
        X = ctx.X
        y = ctx.y

        p = len(parameters_to_vector(network.parameters()))
        n = len(y)
        d3fv2 = torch.zeros(p, dtype=torch.float, device='cuda')
        vector = evec.cuda()
        # for (X, y) in iterate_dataset(dataset, physical_batch_size):
        with torch.enable_grad():
            loss = loss_fn(network(X), y) / n
            grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            grads = torch.autograd.grad(dot, network.parameters(), create_graph=True)
            dot = parameters_to_vector(grads).mul(vector).sum()
            if dot.grad_fn == None:
                grads = torch.zeros_like(parameters_to_vector(grads))
            else:
                grads = torch.autograd.grad(dot, network.parameters(), retain_graph=True)

        return grad_output * parameters_to_vector(grads), None, None, None, None, None

def Lambda(params_dict, network: nn.Module, loss_fn: nn.Module, X: Tensor, y: Tensor) -> torch.Tensor:
    return HessianLargestEigenvalue.apply(params_dict, network, loss_fn, X, y) # type: ignore
