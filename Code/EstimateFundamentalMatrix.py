import numpy as np



def get_normalization_transform(point_set):
    point_set = np.array(point_set)
    x = point_set[:, 0]
    y = point_set[:, 1]
    
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    x_new = x - centroid_x
    y_new = y - centroid_y

    mean_distance = np.mean(np.sqrt(x_new**2 + y_new**2))
    scale = np.sqrt(2)/mean_distance
    
    T = np.eye(3)
    T[0, 0] = scale
    T[0, 2] = -scale*centroid_x
    T[1, 1] = scale
    T[1, 2] = -scale*centroid_y
    
    return T


def estimate_fundamental_matrix(point_set1, point_set2):
    T1 = get_normalization_transform(point_set1)
    T2 = get_normalization_transform(point_set2)
    
    hmg_point_set1  = np.column_stack((point_set1, np.ones(len(point_set1))))
    hmg_point_set2  = np.column_stack((point_set2, np.ones(len(point_set2))))
    
    norm_point_set1 = (T1.dot(hmg_point_set1.T)).T
    norm_point_set2 = (T2.dot(hmg_point_set2.T)).T
    
    # print(norm_point_set1, norm_point_set2)
    A = np.zeros((len(norm_point_set1), 9))
    for i in range(len(norm_point_set1)):
        u, v = norm_point_set1[i][0], norm_point_set1[i][1]
        up, vp = norm_point_set2[i][0], norm_point_set2[i][1]
        A[i] = np.array([up*u, up*v, up, vp*u, vp*v, vp, u, v, 1])
    
    U, S, V_T = np.linalg.svd(A, full_matrices=True)
    F_matrix = V_T.T[:, -1]
    F_matrix = F_matrix.reshape(3, 3)
    
    u, s, v_T = np.linalg.svd(F_matrix)
    s = np.diag(s)
    s[2, 2] = 0
    F_matrix = np.dot(u, np.dot(s, v_T))
    
    F_matrix = np.dot(T2.T, np.dot(F_matrix, T1))
    F_matrix = F_matrix / F_matrix[2, 2]
    
    return F_matrix