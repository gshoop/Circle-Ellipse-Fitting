import numpy as np
import matplotlib.pyplot as plt

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

# Levelnberg-Marquardt Procedure
def lm_fit(a0,b0,R0,lmbd0,F0,X,Y):
    # LM iteration parameters
    alpha = 0.1
    beta = 10
    tol = 10e-6
    inc_k = 1
    norm_ratio = 1
    num_params = 3

    N = len(X)

    ak = a0
    bk = b0
    Rk = R0
    Fk = F0
    lmbd = lmbd0

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

        #L = np.linalg.cholesky(N_lmbd)
        #Lt = L.transpose()
        #z = np.matmul(np.linalg.inv(L),-1*JTg)
        #h = np.linalg.inv(Lt) @ z
        h = np.matmul(np.linalg.inv(N_lmbd),-1*JTg)
        normh = np.linalg.norm(h,2)
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

        mse = np.sum(di**2)/len(di)

    return ak, bk, Rk, mse

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

if __name__ == "__main__":

    circle_fitting()