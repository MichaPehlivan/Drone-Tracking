import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    DURATION = 6.8 #s

    #these detections are directly written down from the plot. #48 datapoints in the designated time window
    detection_ranges = np.array([7.079, 7.28, 7.362, 7.3587, 7.4754, 7.6418, 7.6418, 7.6418, 7.6418, 7.6418, 7.6418, 7.7631, 7.9, 7.92, 8.005, 8.0664, 8.1513, 8.1513, 8.2128, 8.2533, 8.2483, 8.2786, 8.3, 8.4961, 8.4961, 8.4961, 8.4961, 8.4961, 8.4961, 8.4961, 8.7071, 8.77, 8.8359, 9.9208, 9.9208, 9.0625, 9.0625, 9.0625, 9.344])
    detection_angles = np.array([0.0523, 0.0523, 0.0523, 0.0523, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0349, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0.0175, 0,0,0,0,0,0,0,0,0, -0.0175,-0.0175, -0.0342, -0.0342, -0.0523, -0.0523, -0.0523])

    GPS_ranges = np.array([8.2207, 8.2398, 8.2723, 8.2821, 8.2967, 8.3019, 8.2995, 8.3035, 8.3018, 8.3070, 8.2730, 8.2752, 8.2547, 8.2580, 8.2425, 8.2474, 8.2310, 8.2340, 8.2319, 8.2343, 8.2247, 8.2275, 8.2214, 8.2224, 8.2189, 8.2172, 8.2105, 8.2095, 8.2303, 8.2325, 8.2263, 8.2276, 8.2332, 8.2354, 8.2365, 8.2383, 8.2546, 8.2556, 8.2493, 8.2489, 8.2511, 8.2535, 8.2674, 8.2707, 8.2837, 8.2862, 8.2859, 8.2897, 8.2967, 8.3051])
    GPS_angles = np.array([0.0909, 0.0901, 0.0876, 0.0873, 0.0857, 0.0856, 0.0848, 0.0848, 0.0825, 0.0824, 0.0816, 0.0812, 0.0812, 0.0807, 0.0801, 0.0794, 0.0780, 0.0776, 0.0779, 0.0776, 0.0781, 0.0779, 0.0780, 0.0786, 0.0801, 0.0807, 0.0812, 0.0814, 0.0800, 0.0797, 0.0813, 0.0815, 0.0823, 0.0822, 0.0828, 0.0818, 0.0805, 0.0797, 0.0808, 0.0803, 0.0787, 0.0777, 0.0774, 0.0769, 0.0767, 0.0769, 0.0764, 0.0763, 0.0760, 0.0764])

    print("-----GPS------")
    print(f"Mean of the GPS ranges: {np.mean(GPS_ranges)}")
    print(f"Variance of the GPS range: {np.var(GPS_ranges)}\n")

    print(f"Mean of the GPS angles: {np.mean(GPS_angles)}")
    print(f"Variance of the GPS angles: {np.var(GPS_angles)}\n")

    print("-----Detection data------")
    print(f"Mean of the radar range detections: {np.mean(detection_ranges)}")
    print(f"Variance of the radar range: {np.var(detection_ranges)}\n")

    print(f"Mean of the radar angle detections: {np.mean(detection_angles)}")
    print(f"Variance of the radar angles: {np.var(detection_angles)}")


    print((np.deg2rad(4.8)**2)/12)
