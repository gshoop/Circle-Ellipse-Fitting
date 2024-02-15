import numpy as np
import matplotlib.pyplot as plt


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
            

    return ak, bk, Rk

def circle_fitting():
    # Define true circle parameters
    R = 430
    N = 2000
    a = 30
    b = 22
    C = 0

    # Create noisy circle
    mu = 0.0
    var = 500
    k = 0.5
    stdev = np.sqrt(var)
    theta = np.linspace(0,2*np.pi,N)
    theta = np.reshape(theta,(N,1))

    eps_noise_x = k*np.random.normal(mu,stdev,(N,1))
    eps_noise_y = k*np.random.normal(mu,stdev,(N,1))

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
    
    ak,bk,Rk = lm_fit(a0,b0,R0,lmbd,F0,X,Y)

    print(ak)
    print(bk)
    print(Rk)

    alpha = np.linspace(0,2*np.pi,500)

    circle = np.column_stack([ak*np.ones_like(alpha),bk*np.ones_like(alpha)]) + np.column_stack([Rk*np.cos(alpha),Rk*np.sin(alpha)])
    Xt = circle[:,0:1]
    Yt = circle[:,1:]

    axs.plot(Xt,Yt,'r')
    plt.show()
    
if __name__ == "__main__":

    circle_fitting()