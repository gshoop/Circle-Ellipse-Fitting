import numpy as np
import matplotlib.pyplot as plt
import argparse

# Loads uv data from uvd channel file into a numpy array and returns the data
def process_uvd_file(filepath):
    data = np.loadtxt(filepath)
    U = data[:,0]
    V = data[:,2]
    return U,V

# Computing orthogonal distance of a fitted circle
# Circle of the form (x-a)^2 + (y-b)^2 = R^2 is converted to geometric parameters
# to compute the distance of observed data X,Y to the fitted circle with parameters a,b,R
def geometric_distance(a,b,R,X,Y):
    # Convert algebraic parameters a,b,R to geometric parameters A,B,C,D
    A = 1/(2*R)
    B = -2*A*a
    C = -2*A*b
    D = (B**2 + C**2 -1)/(4*A)

    Pi = A*(X**2 + Y**2) + B*X + C*Y + D
    
    di = 2*Pi/(1+np.sqrt(1 + 4*A*Pi))

    return di

# Levenberg-Marquardt Procedure for fitting circles
def lm_fit(a0,b0,R0,X,Y):
    # LM iteration parameters
    alpha = 0.1
    beta = 10
    tol = 10e-6
    inc_k = 1
    norm_ratio = 1
    num_params = 3
    lmbd = 10e-3
    N = len(X)

    # Initialize iterated parameters
    ak = a0
    bk = b0
    Rk = R0
    Fk = np.sum((np.sqrt((X-a0)**2+(Y-b0)**2) - R0)**2)

    # Iterate while ||h||/Rk > tolerance
    while norm_ratio > tol:
        if inc_k == 1:
            r = np.sqrt((X-ak)**2+(Y-bk)**2)
            u = (X-ak)/r
            v = (Y-bk)/r
            uu_mean = 1/N * np.sum(u**2)
            vv_mean = 1/N * np.sum(v**2)
            uv_mean = 1/N * np.sum(u*v)
            u_mean = 1/N * np.sum(u)
            v_mean = 1/N * np.sum(v)

            N_mat = N*np.array([[uu_mean,uv_mean,u_mean],[uv_mean,vv_mean,v_mean],[u_mean,v_mean,1]])
            J = np.column_stack([-1*u,-1*v,-1*np.ones_like(u)])
            g = r-Rk
            # Solve linear system Nh = -J^T g for h with Cholesky decomposition
            JTg = J.transpose() @ g
        I = np.identity(num_params)*lmbd
        N_lmbd = N_mat + I

        # Cholesky decomposition (could be implemented if direct inversion is not fast enough)
        #L = np.linalg.cholesky(N_lmbd)
        #Lt = L.transpose()
        #z = np.matmul(np.linalg.inv(L),-1*JTg)
        #h = np.linalg.inv(Lt) @ z
        h = np.matmul(np.linalg.inv(N_lmbd),-1*JTg)
        norm_ratio = np.linalg.norm(h,2)/Rk

        if norm_ratio > tol:
            # Update a, b, R with h
            ak_1 = ak + h[0].squeeze()
            bk_1 = bk + h[1].squeeze()
            Rk_1 = Rk + h[2].squeeze()
            Fk_1 = np.sum((np.sqrt((X-ak_1)**2+(Y-bk_1)**2) - Rk_1)**2)

            if Fk_1 >= Fk or Rk_1 <= 0:
                lmbd = beta*lmbd
                inc_k = 0
            else:
                lmbd = alpha*lmbd
                inc_k = 1

                ak = ak_1
                bk = bk_1
                Rk = Rk_1
                Fk = Fk_1

        di = geometric_distance(ak,bk,Rk,X,Y)

        #residual = np.sqrt(np.sum(di**2))
        mean = np.sum(np.abs(di))/len(di)
        mse = np.sum(di**2)/len(di)

    return ak, bk, Rk, mse, mean

def circle_fit(U,V):
    # Formulate least squares problem ||Ax-b||^2
    A = np.column_stack((U**2 + V**2,U,V))
    b = np.ones_like(U)

    solution, residuals, rank, singular_value = np.linalg.lstsq(A,b,rcond=None)

    return solution, residuals.squeeze()

def circle_fitting():
    # Define true circle parameters
    R = 40
    N = 20000
    a = 0
    b = 0
    C = 0

    # Create noisy circle
    mu = 0.0
    var = 25
    k = 0.5
    stdev = np.sqrt(var)
    theta = np.random.normal(loc=np.linspace(0,2*np.pi,N),scale=stdev)
    theta = np.reshape(theta,(N,1))

    eps_noise_x = k*np.random.normal(mu,stdev,(N,1))
    eps_noise_y = k*np.random.normal(mu,stdev,(N,1))
    print(eps_noise_x)

    noisy_circle = np.column_stack([a + R*np.cos(theta) + eps_noise_x,b + R*np.sin(theta) + eps_noise_y])
    X = noisy_circle[:,0]
    Y = noisy_circle[:,1]

    # Plot data samples
    fig, axs = plt.subplots()
    axs.scatter(X,Y,s=1.0)
    axs.grid(True)
    axs.set_aspect('equal','datalim')

    # Initialize guesses for parameters
    a0 = 1
    b0 = 2
    R0 = 21
    lmbd = 10e-3

    #Compute f0
    F0 = np.sum((np.sqrt((X-a0)**2+(Y-b0)**2) - R0)**2)
    
    ak,bk,Rk,res = lm_fit(a0,b0,R0,lmbd,F0,X,Y)
    x,err = circle_fit(X,Y)

    A = x[0]
    B = x[1]
    C = x[2]
    D = -1

    print(f'A: {A}')
    print(f'B: {B}')
    print(f'C: {C}')



    ac = -B/(2*A)
    bc = -C/(2*A)
    rcsq = (B**2 + C**2 -4*A*D)/(4*A**2)
    rc = np.sqrt(rcsq)

    constr = B**2 + C**2 - 4*A*D

    print(f'constr: {constr}')
    print(f'a: {ak}')
    print(f'b: {bk}')
    print(f'R: {Rk}')
    print(f'res: {res}')
    print(f'err: {err}')

    alpha = np.linspace(0,2*np.pi,500)

    circle = np.column_stack([ak*np.ones_like(alpha),bk*np.ones_like(alpha)]) + np.column_stack([Rk*np.cos(alpha),Rk*np.sin(alpha)])
    circ = np.column_stack([ac*np.ones_like(alpha),bc*np.ones_like(alpha)]) + np.column_stack([rc*np.cos(alpha),rc*np.sin(alpha)])
    Xt = circle[:,0]
    Yt = circle[:,1]

    Xc = circ[:,0]
    Yc = circ[:,1]

    axs.plot(Xt,Yt,'r')
    axs.plot(Xc,Yc,'k--')
    plt.show()

def ellipse_fit(U,V):
    # Formulate least squares problem ||Ax-b||^2
    A = np.column_stack((U**2, U*V, V**2,U,V))
    b = np.ones_like(U)
    solution, residual, rank, singular_values = np.linalg.lstsq(A,b,rcond=None)

    y = np.matmul(A,solution)
    err = y-b

    return solution, residual.squeeze(),err

def parse_input():
    parser = argparse.ArgumentParser(description="Process .uvd files in a folder and plot data.")
    parser.add_argument('input_folder', help="Path to the folder containting .txt file of corrected UV Data")

    args = parser.parse_args()

    return args.input_folder

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return (x0, y0, ap, bp, e, phi)

def get_ellipse_pts(coeffs, npts=500, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """
    params = cart_to_pol(coeffs)
    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

if __name__ == "__main__":

    input_file = parse_input()
    U,V = process_uvd_file(input_file)

    a0 = 0.1
    b0 = 0.2
    R0 = 550

    # Ellipse fit
    x,res,err = ellipse_fit(U,V)

    coeffs = []
    for item in x:
        coeffs.append(item)
    coeffs.append(-1.0)

    distance = np.sum(np.abs(err))/len(err)
    ind = np.argmax(err)
    print(f'{U[ind]},{V[ind]}')
    x,y = (U[ind],V[ind])
    E = coeffs[0]*x**2 + coeffs[1]*x*y + coeffs[2]*y**2 + coeffs[3]*x + coeffs[4]*y
    print(E)
    print(f'distance: {distance:.3f}')

    #x0,y0,ap,bp,e,phi = cart_to_pol(coeffs)
    #Xe,Ye = get_ellipse_pts((x0,y0,ap,bp,e,phi))
    Xe,Ye = get_ellipse_pts(coeffs)
    # Circle fit
    a,b,R,mse,di = lm_fit(a0,b0,R0,U,V)
    alpha = np.linspace(0,2*np.pi,500)

    circle = np.column_stack([a + R*np.cos(alpha),b + R*np.sin(alpha)])
    X = circle[:,0]
    Y = circle[:,1]

    fig,axs = plt.subplots()

    axs.scatter(U,V,s=0.9,label='Corrected UV Data')
    axs.plot(X,Y,'k--',label="LM-Circle Fit",linewidth=0.8)
    axs.plot(Xe,Ye,'r-',label="Ellipse Fit",linewidth=0.8)
    axs.scatter(U[ind],V[ind],s=8.0)
    axs.text(-200,0,f'MSE = {mse:.3E}',color='black')
    axs.text(-200,100,f'Average distance = {di:.3E}',color='black')
    axs.set(xlabel="U (adc)", ylabel="V (adc)")
    axs.set_aspect('equal','datalim')
    axs.grid(True)
    axs.legend(loc='upper right')
    plt.show()


    #circle_fitting()