import numpy as np

def filter():
    input_image = np.array([[1, 2, 6, 7], \
                            [2, 3, 5, 4], \
                            [8, 7, 8, 3], \
                            [1, 1, 2, 2]])

    input_image2 = np.array([[1, 6, 7, 3], \
                            [5, 8, 2, 3], \
                            [2, 5, 7, 8], \
                            [9, 1, 8, 2]])

    input_image = np.array([[2, 4, 6, 7], \
                            [5, 3, 1, 9], \
                            [3, 2, 7, 8], \
                            [9, 2, 1, 8]])

    median_filtered = np.zeros((2, 2))

    for y in range(2):
        for x in range(2):
            window = input_image[y:y + 3, x:x + 3]
            print(window)
            print(f'meidan: {np.median(window.flatten())}')
            median_filtered[y, x] = np.median(window.flatten())
            print()
            median_filtered

    print(input_image)
    print(median_filtered)

def adaptive_threshold():
    # values = [4, 16, 64, 8, 4, 32, 16, 64, 32, 4, 4, 4, 32, 64, 4, 64]
    # values = [32, 8, 32, 8, 32, 128, 16, 64, 128, 128, 8, 8, 16, 8, 64, 128]
    values = [16, 128, 255, 255, 128, 16, 64, 255, 128, 4, 16, 16, 16, 255, 128, 255]
    values.sort()
    print(values)
    theta = np.mean(values)
    for i in range(5):
        print(f'Theta {i} = {theta}')
        u_below = np.mean([x for x in values if x <= theta])
        u_above = np.mean([x for x in values if x > theta])
        print(f'u_below={u_below}, u_above={u_above}')
        theta = (u_above + u_below) / 2


def transformation_matrix_2d(delta_x = 0, delta_y = 0):
    homogeneous_transformation_matrix = [
        [1, 0, delta_x],
        [0, 1, delta_y],
        [0, 0, 1]
    ]
    return np.array(homogeneous_transformation_matrix)

def rotation_matrix_2d(theta = 0): #theta is in radians
    np.cos()


if __name__=="__main__":
    # filter()
    adaptive_threshold()
    # values = [255, 8, 8, 64, 128, 64, 16, 128, 64, 8, 32, 32, 128, 64, 128, 255]
    # values.sort()
    # print(values)

