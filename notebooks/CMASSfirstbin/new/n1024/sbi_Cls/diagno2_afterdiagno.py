import numpy as np

w_theta = np.array([-0.5333,-0.5924,-0.6385,-0.6853,-0.7496,-0.8176,-0.8787,
                    -0.9463,-1.0516,-1.1516,-1.2965,-1.4574,-1.6724,-1.9193,
                    -2.2314,-2.6074,-3.0738,-3.5903,-4.1039,-4.4645])
w_nu    = np.array([ 2.5816, 2.8196, 2.9811, 3.1407, 3.3513, 3.5585, 3.7038,
                     3.8602, 4.1641, 4.4060, 4.8495, 5.3233, 6.0159, 6.8450,
                     7.9266, 9.3330,11.135, 13.272, 15.547, 17.485])

F = np.array([
    [np.dot(w_theta, w_theta), np.dot(w_theta, w_nu)],
    [np.dot(w_theta, w_nu),    np.dot(w_nu,    w_nu)],
])
print("Fisher matrix:\n", F)
print("det(F):", np.linalg.det(F))

C = np.linalg.inv(F)
print("sigma_theta (marginal):", np.sqrt(C[0,0]))
print("sigma_nu    (marginal):", np.sqrt(C[1,1]))
print("correlation r:", C[0,1]/np.sqrt(C[0,0]*C[1,1]))

print("\nnu prior width: 0.3  vs  sigma_nu:", np.sqrt(C[1,1]))
print("=> nu posterior is prior-dominated (flat). Expected.")
