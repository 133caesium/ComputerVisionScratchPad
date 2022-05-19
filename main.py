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
    homogeneous_rotation_matrix = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ]
    return np.array(homogeneous_rotation_matrix)

def scale_matrix_2d(scale_x = 1, scale_y = 1):
    homogeneous_scale_matrix = [
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ]
    return np.array(homogeneous_scale_matrix)

def shear_matrix_2d(h_x=1, h_y=1):
    homogeneous_shear_matrix = [
        [1, h_x, 0],
        [h_y, 1, 0],
        [0, 0, 1]
    ]
    return np.array(homogeneous_shear_matrix)

def inverse_matrix(m):
    return np.linagl.inv(m)

class BarrelDistortionObject:
    def __init__(self, pixels_x=3000, pixels_y=2250, sensor_width=6.0, sensor_height=4.5):
        self.sensor_width_px = pixels_x
        self.sensor_height_px = pixels_y
        self.sensor_width_mm = sensor_width
        self.sensor_height_mm = sensor_height
        self.pixel_width = sensor_width/pixels_x
        self.pixel_height = sensor_height/pixels_y
        print(f"pixel width: {self.pixel_width}, pixel height: {self.pixel_height}")


    def calculate_k1_in_mm(self):
        # p**2 = xd**2 + yd**2
        # xu = xd * (1 + k1 * p**2)
        # xu / xd = 1 + k1 * p**2
        # xu / xd -1 = k1 * p**2
        # (xu / xd -1) / p**2 = k1
        # k1 = (xu/xd -1)/p**2
        # k1 = (xu/xd -1)/(xd**2+yd**2)
        xu = self.sensor_width_mm/2
        xd = xu + self.pixel_width
        yd = self.sensor_height_mm/2-self.pixel_width
        k1 = (xu/xd -1)/(xd**2+yd**2)
        return k1

    def calculate_something(self):
        pass

def compute_xuyu_given_xdyd(xd_px, yd_px, k1_um):
    pixels_wide = 5000
    pixels_high = 3750
    sensor_wide = 5.0
    sensor_high = 3.75
    pixel_width = sensor_wide/pixels_wide
    xd = (xd_px-pixels_wide/2)*pixel_width
    yd = (yd_px - pixels_high / 2) * pixel_width

    # k1 = (xu/xd -1)/(xd**2+yd**2)
    k1 = k1_um * 0.001
    p2 = ((xd/2))**2 + ((yd/2))**2
    xu = xd * (1 + k1 * p2) / pixel_width + pixels_wide/2
    yu = yd * (1 + k1 * p2) / pixel_width + pixels_high / 2
    return xu, yu

def compute_xuyu_given_xdyd_2(xd_px, yd_px, k1_um):
    pixels_wide = 3000
    pixels_high = 2250
    sensor_wide = 6.0
    sensor_high = 4.5

    pixel_width = sensor_wide / pixels_wide
    xd = (xd_px - pixels_wide / 2) * pixel_width
    yd = (yd_px - pixels_high / 2) * pixel_width

    k1 = k1_um * 0.001
    p2 = ((xd/2))**2 + ((yd/2))**2
    xu = xd * (1 + k1 * p2) / pixel_width + pixels_wide / 2
    yu = yd * (1 + k1 * p2) / pixel_width + pixels_high / 2
    return xu, yu

def compute_k1i(Xd, Xu, Yd, Yu):
    delta_i = np.sqrt((Xd-Xu)**2+(Yd-Yu)**2)
    p_i = np.sqrt(Xd**2+Yd**2)
    k1i = delta_i / p_i**3
    return k1i


if __name__=="__main__":
    # filter()
    # adaptive_threshold()
    # values = [255, 8, 8, 64, 128, 64, 16, 128, 64, 8, 32, 32, 128, 64, 128, 255]
    # values.sort()
    # print(values)
    # print(np.dot(transformation_matrix_2d(3,2),rotation_matrix_2d(np.pi/4)))
    # m = np.dot(scale_matrix_2d(2, 10),np.dot(transformation_matrix_2d(9, -3), rotation_matrix_2d(np.pi / 6)))
    # print(m)
    # print(np.dot(m,np.transpose(np.array([1,1,1]))))

    m = np.dot(transformation_matrix_2d(8, 2), rotation_matrix_2d(np.pi / 6))
    # print(np.linalg.det(m))
    print(np.linalg.inv(m))

    test = BarrelDistortionObject()
    print(test.calculate_k1_in_mm())
    print(f'compute_xuyu_given_xdyd(3571, 685, -1.2) = {compute_xuyu_given_xdyd(3571, 685, -1.2)}')
    print(f'compute_xuyu_given_xdyd(5000, 0, -1.2) = {compute_xuyu_given_xdyd(5000, 0, -1.2)}')
    print(f'compute_xuyu_given_xdyd_2(x, 0, -1.2) = {compute_xuyu_given_xdyd_2(3000, 0, -0.44)}')
    print(f'Using slide 10 lecture content k1 = {compute_k1i(6.5/2,6.5/2+0.002,4.5/2, 4.5/2+0.002)}')

    xd = 2351-1500
    xu = xd-1
    yd = 1155-1125
    k1 = (xu / xd - 1) / (xd ** 2 + yd ** 2)
    print(k1)