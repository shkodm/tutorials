# following the FEniCS tutorial book, section 5.5.2 (https://fenicsproject.org/pub/tutorial/pdf/fenics-tutorial-vol1.pdf)

from dolfin import Expression, interpolate, grad, dot, Point, RectangleMesh, FunctionSpace, VectorFunctionSpace, project, FacetNormal


def fluxes_from_temperature_full_domain(u, V, mesh):
    """
    compute flux from weak form (see FEniCS tutorial book, section 5.5.2)
    :param u: known temperature field
    :param V: function space
    :param mesh: the underlying mesh
    :param alpha: heat conduction coefficient
    :return:
    """
    degree = V.ufl_element().degree()
    W = VectorFunctionSpace(mesh, 'P', degree)
    grad_u_x, grad_u_y = project(grad(u), W).split()
    return grad_u_x  # todo: this is not general! In the end we will need the normal flux.


alpha = 3  # parameter alpha
beta = 1.3  # parameter beta

y_bottom, y_top = 0, 1
x_left, x_right = 0, 2
x_coupling = 1.5  # x coordinate of coupling interface

p0 = Point(x_left, y_bottom)
p1 = Point(x_coupling, y_top)
nx = 15
ny = 10

mesh = RectangleMesh(p0, p1, nx, ny)
V = FunctionSpace(mesh, 'P', 2)

u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
u = interpolate(u_D, V)

grad_u_x = fluxes_from_temperature_full_domain(u, V, mesh, alpha)

print(grad_u_x(1.5, .5))