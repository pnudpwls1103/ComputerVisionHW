import numpy as np
import matplotlib.pyplot as plt

# img1 = plt.imread('./data/warrior_a.jpg')
# img2 = plt.imread('./data/warrior_b.jpg')

# cor1 = np.load("./data/warrior_a.npy")
# cor2 = np.load("./data/warrior_b.npy")

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    ### YOUR CODE BEGINS HERE
    t1 = np.transpose(x1)
    t2 = np.transpose(x2)

    # build matrix for equations in Page 51
    matrix_a = []
    for e1, e2 in zip(t1, t2):
        matrix_a.append([e1[0]*e2[0], e1[1]*e2[0], e2[0], e1[0]*e2[1], e1[1]*e2[1], e2[1], e1[0], e1[1], 1])
    matrix_a = np.asarray(matrix_a)

    # compute the solution in Page 51
    ATA = matrix_a.T @ matrix_a
    eigvals, eigvecs = np.linalg.eig(ATA)
    v = eigvecs[:, np.argmin(eigvals)]
    F = v.reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    sigma = np.diag(S)
    sigma[2][2] = 0

    F = U @ sigma @ V
    ### YOUR CODE ENDS HERE
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    U,S,V = np.linalg.svd(F)    # solve Fe1 = 0 by SVD
    e1 = V[-1]
    e1 = e1/e1[2]

    U,S,V = np.linalg.svd(F.T) # solve F_2e2 = 0 by SVD
    e2 = V[-1]
    e2 = e2/e2[2]
    ### YOUR CODE ENDS HERE
    
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE
    def cal_ab(x1, y1, x2, y2):
        a = (y2 - y1) / (x2 - x1)
        b = (x2*y1 - x1*y2) / (x2 - x1)
        return a, b
    
    cor1 = np.transpose(cor1)
    cor2 = np.transpose(cor2)
    
    size = img1.shape
    fig, axes = plt.subplots(1, 2, figsize=(30, 30))
    for c1, c2 in zip(cor1, cor2):
        color = (np.random.random(), np.random.random(), np.random.random())

        x1, y1 = e1[:2]
        x2, y2 = c1[:2]
        a, b = cal_ab(x1, y1, x2, y2)
        x = np.array([0, size[1]])
        axes[0].plot(x, a*x+b, c = color)
        axes[0].plot(x2, y2, 'o', c = color)

        x1, y1 = e2[:2]
        x2, y2 = c2[:2]
        a, b = cal_ab(x1, y1, x2, y2)
        x = np.array([0, size[1]])
        axes[1].plot(x, a*x+b, c = color)
        axes[1].plot(x2, y2, 'o', c = color)

    axes[0].imshow(img1)
    axes[1].imshow(img2)
    
    fig.show()
    input('Press any key to exit the program')

    # fig.savefig('warrior.png')
    fig.savefig('graffit.png')
    ### YOUR CODE ENDS HERE
    return

draw_epipolar_lines(img1, img2, cor1, cor2)