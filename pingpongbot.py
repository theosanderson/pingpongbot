import cv2
import fire, time
import numpy as np
import json
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt

from xarm.wrapper import XArmAPI

ip = "192.168.0.207"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(7)
arm.set_state(state=0)
arm.set_tcp_jerk (10000)
arm.set_joint_jerk (500, is_radian = True)
arm.set_tcp_maxacc(50000)
arm.save_conf ()
arm.set_mode(7)
arm.set_state(0)
speed1=500
mvacc1=10000


arm.set_position(x=-300, y=20, z=500, roll=177.9, pitch=1.1, yaw=87, speed=1000, wait=True, is_radian= False)
arm.set_mode(7)
from threading import Thread

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.status = False
        self.frame = None
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.new_frame = False

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                self.new_frame = True
            time.sleep(.01)
    
    def get_frame(self):
        # Return the current frame
        self.new_frame = False
        return self.status, self.frame


xtable = 2.74
ytable = 1.525

webcam_1 = 3
webcam_2 = 0




class Tracker:
    
    def __init__(self, c1_calibration, c2_calibration):
        p1 = [0, 0, 0, 1]
        p2 = [xtable /2, 0, 0.15, 1]
        p3 = [xtable, 0, 0, 1]
        p4 = [xtable, ytable, 0, 1]
        p5 = [xtable / 2,ytable, 0.15, 1]
        p6 = [0, ytable, 0, 1]
        self.c3d = np.transpose(np.array([p1, p2, p3, p4, p5, p6]))

        self.pc1 = np.transpose(self.process_calibration(c1_calibration))
        self.pc2 = np.transpose(self.process_calibration(c2_calibration))
        self.P1 = calc_P(self.c3d, self.pc1)
        self.P2 = calc_P(self.c3d, self.pc2)
        [r1, q1] = rq(self.P1)
        [r2, q2] = rq(self.P2)
        self.K1 = r1
        self.K2 = r2
        self.A1 = q1
        self.A2 = q2
        self.P1norm = np.matmul(np.linalg.inv(self.K1), self.P1)
        self.P2norm = np.matmul(np.linalg.inv(self.K2), self.P2)

    def process_calibration(self, calibration):
        pts = [[p[0], calibration['h'] - p[1], 1] for p in calibration['points']]
        return np.array(pts)

    def calc_3d_point(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Finds a 3D point from two 2D points

        Args:
            x1 (np.ndarray): Point of shape (3,1) with last value equal to 1
            x2 (np.ndarray): Point of shape (3,1) with last value equal to 1

        Returns:
            np.ndarray: 3D point of shape (3,)
        """
        x1norm = np.matmul(np.linalg.inv(self.K1), x1)
        x2norm = np.matmul(np.linalg.inv(self.K2), x2)
        M = np.zeros([6, 6])
        M[0:3, 0:4] = self.P1norm
        M[3:6, 0:4] = self.P2norm
        M[0:3, 4] = -x1norm
        M[3:6, 5] = -x2norm
        [_, _, V] = np.linalg.svd(M)
        v = V[5, :]
        X = pflat(np.reshape(v[0:4], [4, 1]))
        return np.reshape(
            X[0:3],
            [
                3,
            ],
        )

        

def get_cameras():
    cap1 = cv2.VideoCapture(webcam_1)
    cap2 = cv2.VideoCapture(webcam_2)
    return cap1, cap2



def calibrate_one(cap):
   
    _, frame = cap.read()
    h, w, c = frame.shape
    h1, _, c1 = frame.shape

    class CoordinateStore:
        def __init__(self):
            self.points = []

        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                self.points.append((x, y))


    # instantiate class
    coordinateStore1 = CoordinateStore()
    # Create a black image, a window and bind the function to window
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", coordinateStore1.select_point)
    while True:
        cv2.imshow("image", frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return {"points": coordinateStore1.points, "h": h, "w": w, "c": c}

def calibrate():
    cap1, cap2 = get_cameras()
    print("Window kitchen, then net, then window non-kitchen, then non-window non-kitchen, then net, then non-window kitchen.")
    print("Then press esc to finish.")
    c1 = calibrate_one(cap1)
    print("Now do the mask")
    c1['mask'] = calibrate_one(cap1)
    print("Window kitchen, then net, then window non-kitchen, then non-window non-kitchen, then net, then non-window kitchen.")
    print("Then press esc to finish.")
    c2 = calibrate_one(cap2)
    print("Now do the mask")
    c2['mask'] = calibrate_one(cap2)
    all_points = {'c1': c1, 'c2': c2}
    with open('calibration_points.json', 'w') as f:
        json.dump(all_points, f)


def check_calibration():
    with open('calibration_points.json') as f:
        all_points = json.load(f)
    # capture images from each webcam and display the points on top
    cap1, cap2 = get_cameras()
    ret1, frame1 = cap1.read()
    # display c1 on top of frame1
    for point in all_points['c1']['points']:
        cv2.circle(frame1, point, 3, (255, 0, 0), -1)
    ret2, frame2 = cap2.read()
    # display c2 on top of frame2
    for point in all_points['c2']['points']:
        cv2.circle(frame2, point, 3, (255, 0, 0), -1)
        cv2.circle(frame2, point, 3, (255, 0, 0), -1)
    frame = np.concatenate((frame1, frame2), axis=1)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)


def calc_P(p3d: np.ndarray, p2d: np.ndarray) -> np.ndarray:
    """Calculates camera matrix from a set of 6 point correspondences

    Args:
        p3d (np.ndarray): 3D known points of shape (4,6) (last value is 1)
        p2d (np.ndarray): 2D points of shape (3,6) (last value is 1)

    Returns:
        np.ndarray: Camera matrix
    """
    npoints = p2d.shape[1]
    mean = np.mean(p2d, 1)
    std = np.std(p2d, axis=1)
    N = np.array(
        [
            [1 / std[0], 0, -mean[0] / std[0]],
            [0, 1 / std[1], -mean[1] / std[1]],
            [0, 0, 1],
        ]
    )
    p2dnorm = np.matmul(N, p2d)
    M = np.zeros([3 * npoints, 12 + npoints])
    for i in range(npoints):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i : 3 * i + 3, 12 + i] = -p2dnorm[:, i]
    [U, S, V] = svd(M)
    v = V[V.shape[0] - 1, :]
    P = np.reshape(v[0:12], [3, 4])
    testsign = np.matmul(P, p3d[:, 1])
    if testsign[2] < 0:
        P = -P
        print("changed sign of P")
    P = np.matmul(np.linalg.inv(N), P)
    return P


def rq(a: np.ndarray) -> tuple:
    """RQ-factorization

    Args:
        a (np.ndarray): Original matrix.

    Returns:
        tuple: (r: np.ndarray, q: np.ndarray) rq=a
    """
    [m, n] = a.shape
    e = np.eye(m)
    p = np.fliplr(e)
    [q0, r0] = qr(np.matmul(p, np.matmul(np.transpose(a[:, 0:m]), p)))
    r = np.matmul(p, np.matmul(np.transpose(r0), p))
    q = np.matmul(p, np.matmul(np.transpose(q0), p))
    fix = np.diag(np.sign(np.diag(r)))
    r = np.matmul(r, fix)
    q = np.matmul(fix, q)
    if n > m:
        q = np.concatenate((q, np.matmul(np.linalg.inv(r), a[:, m:n])), axis=1)
    return r, q

def read_calibration_points():
    with open('calibration_points.json') as f:
        all_points = json.load(f)
    return all_points

def getSinglePoint(cap):
    # display a frame from the cap and get the user to double click on a point and capture that position
    _, frame = cap.read()

    cv2.namedWindow("image")
    class CoordinateStore():
        def __init__(self):
            self.points = []
        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                self.points = [x, y]
                print(self.points)
                cv2.destroyAllWindows()
    coordinateStore1 = CoordinateStore()
    cv2.setMouseCallback("image", coordinateStore1.select_point)
    cv2.imshow("image", frame)
    while coordinateStore1.points == []:
        cv2.waitKey(20)
        
        

 
    return coordinateStore1.points




def test_2d_point():
    # Click the same point on each webcam, and print the calculated 3D point
    cap1, cap2 = get_cameras()
    calibration_points = read_calibration_points()
    singlePointC1 = getSinglePoint(cap1) #[502, 204] # 
    singlePointC2 = getSinglePoint(cap2)#[141, 286] #
    tracker = Tracker(calibration_points['c1'], calibration_points['c2'])
    singlePointC1 = [singlePointC1[0], calibration_points['c1']['h'] -singlePointC1[1], 1]
    singlePointC2 = [singlePointC2[0], calibration_points['c2']['h'] -singlePointC2[1], 1]
    three_d_point = tracker.calc_3d_point(singlePointC1, singlePointC2)
    print(three_d_point)

def debug_hsv():
    cam1, cam2 = get_cameras()
    relevant_cam = cam2
    # get an image, convert to HSV
    _, frame = relevant_cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # when a pixel is clicked, print the HSV value of that pixel
    cv2.namedWindow("image")
    class CoordinateStore():
        def __init__(self):
            self.points = []
        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                
                print(hsv[y, x])
                
    coordinateStore1 = CoordinateStore()
    cv2.setMouseCallback("image", coordinateStore1.select_point)
    cv2.imshow("image", frame)
    while True:
        cv2.waitKey(20)
        if coordinateStore1.points != []:
            break

def find_ball(frame, mask_polygon = None):
    if mask_polygon:
        #polygon_of_interest =[[12, 242], [89, 441], [598, 286], [436, 153]]
        polygon_of_interest = [np.array(point) for point in mask_polygon]
        polygon_of_interest = np.array(polygon_of_interest)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [polygon_of_interest], [255, 255, 255])
        # display the mask
       # cv2.imshow("mask2", mask)
        masked_frame = cv2.bitwise_and(frame, mask)
        frame = masked_frame




    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # find the red ball
    lower_red = np.array([160, 130, 80])
    upper_red = np.array([185, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        cnt = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        return center, radius, mask
    else:
        return None, None, mask



def ball_test():
    cap1, cap2 = get_cameras()
    frame1 = cap1.read()[1]
    # find the ball and annotate it on the frame
    center, radius = find_ball(frame1, mask_polygon=[[[12, 242], [89, 441], [598, 286], [436, 153]]])
    if center is not None:
        cv2.circle(frame1, center, radius, (0, 255, 255), 2)
    cv2.imshow('frame1', frame1)
    cv2.waitKey(0)
    




def pflat(x: np.ndarray) -> np.ndarray:
    """Pointwise division with last coordinate

    Args:
        x (np.ndarray): Array to apply pflat to usually of shape (3,N).

    Returns:
        np.ndarray: Result.
    """
    y = np.copy(x)
    for i in range(x.shape[1]):
        y[:, i] = y[:, i] / y[x.shape[0] - 1, i]
    return y


def define_mask():
    # Display an image, and capture 4 points from the user
    # The points are the corners of the mask

    cam1, cam2 = get_cameras()
    relevant_camera= cam2
    _, frame = relevant_camera.read()
    cv2.namedWindow("image")
    class CoordinateStore():
        def __init__(self):
            self.points = []
        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                self.points.append([x, y])
                print(self.points)
                
    coordinateStore1 = CoordinateStore()
    cv2.setMouseCallback("image", coordinateStore1.select_point)
    cv2.imshow("image", frame)
    while len(coordinateStore1.points) < 5:
        cv2.imshow("image", frame)
        cv2.waitKey(20)

    print(coordinateStore1.points)

import random
import os
import time
from dataclasses import dataclass

@dataclass
class TimeAndPoint:
    time: float
    point: np.ndarray

class TrajectoryPrediction():
    max_points = 10
    data = []
    def add_point(self, point):
        self.data.append(TimeAndPoint(time=time.time(), point=point))
        if len(self.data) > self.max_points:
            self.data.pop(0)
    def update_velocities(self):
        for i in range(len(self.data) - 1):
            self.data[i].velocity = (self.data[i+1].point - self.data[i].point) / (self.data[i+1].time - self.data[i].time)
    def update_accelerations(self):
        for i in range(len(self.data) - 2):
            self.data[i].acceleration = (self.data[i+1].velocity - self.data[i].velocity) / (self.data[i+1].time - self.data[i].time)
    
    def identify_bounces(self):
        bounces = []
        for i in range(len(self.data) - 2):
            if self.data[i].acceleration[2] > 0 and self.data[i+1].acceleration[2] < 0:
                bounces.append(i)
        return bounces
    
    def identify_where_to_intercept(self, time):
        """Predict into the future, based on the current trajectory. Account for bounces"""
        self.update_velocities()
        self.update_accelerations()
        bounces = self.identify_bounces()
        # get the section after the last bounce
        if len(bounces) > 0:
            section = self.data[bounces[-1]+1:]
        else:
            section = self.data
        
        # average over the last 3 velocities and accelerations
        velocity = np.mean([point.velocity for point in section[-3:]], axis=0)
        acceleration = np.mean([point.acceleration for point in section[-3:]], axis=0)
        # average over the last 3 positions
        position = np.mean([point.point for point in section[-3:]], axis=0)
        # average over last 3 times
        time = np.mean([point.time for point in section[-3:]], axis=0)

        def f(t):
            return position + velocity * t + 0.5 * acceleration * t**2
        
        min_z = 0.1
        max_z = 0.4
        min_x = 0.3
        max_x = 0.4
        min_y=0
        max_y=ytable
        # iterate over the next 3 secs and identify a position where the constraints are satisfied
        for t in np.linspace(0, 3, 100):
            point = f(t)
            if min_z < point[2] < max_z and min_x < point[0] < max_x and min_y < point[1] < max_y:
                return point

        





def view_all_cameras():
    # Get a frame from every camera available
    n=0
    frames = []
    while True:
        cam = cv2.VideoCapture(n)
        _, frame = cam.read()
        if _:
            frames.append(frame)
        else:
            # append empty frame if no camera is available
            frames.append(np.zeros((480, 640, 3)))
        if n > 9:
            break
        n += 1
    # display all the frames
    for i, frame in enumerate(frames):
        print(i)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)





def main():
    print('hi')
    id = random.randint(0, 10000)
    fo = open("streams/" + str(id) + ".txt", "wt")
    os.mkdir(f"streams_img/{id}")




    """Capture from webcam 1 and 2 at the same time and display them side by side."""

    calibration_points = read_calibration_points()
    tracker = Tracker(calibration_points['c1'], calibration_points['c2'])
    trajectory = TrajectoryPrediction()


    widget1 = VideoStreamWidget(webcam_1)
    widget2 = VideoStreamWidget(webcam_2)
    n= 0
    print("start")
    old_pos = [0,0,0]
    while True:
        time.sleep(0.01)
        if(widget1.new_frame or widget2.new_frame) and widget1.status and widget2.status:
            ret1, frame1 = widget1.get_frame()
            ret2, frame2 = widget2.get_frame()
            ball_pos_1_center, radius1, mask1 = find_ball(frame1, mask_polygon=calibration_points['c1']['mask']['points'])  
            ball_pos_2_center, radius2, mask2 = find_ball(frame2, mask_polygon=calibration_points['c2']['mask']['points'])
        
            if ball_pos_1_center is not None:
                cv2.circle(frame1, np.array(ball_pos_1_center), radius1, (0, 255, 255), 2)
            if ball_pos_2_center is not None:
                cv2.circle(frame2, np.array(ball_pos_2_center), radius2, (0, 255, 255), 2)

            if (ball_pos_1_center is not None) and (ball_pos_2_center is not None):
                singlePointC1 = [ball_pos_1_center[0], calibration_points['c1']['h'] -ball_pos_1_center[1], 1]
                singlePointC2 = [ball_pos_2_center[0], calibration_points['c2']['h'] -ball_pos_2_center[1], 1]
                three_d_point = tracker.calc_3d_point(singlePointC1, singlePointC2)
                trajectory.add_point(three_d_point)
                print(three_d_point)
                if (three_d_point[0]<2.2):
                    ytarget = 1000*ytable/2 - 1000*three_d_point[1]-100
                    y_min = -300
                    y_max =  300
                    if ytarget < y_min:
                        ytarget = y_min
                    if ytarget > y_max:
                        ytarget = y_max
                    print("Y target: ", ytarget)
                    ztarget = 1000*three_d_point[2]+120
                    z_min = 400
                    z_max =  600
                    if ztarget < z_min:
                        ztarget = z_min
                    if ztarget > z_max:
                        ztarget = z_max
                    xtarget = -300
                    newpos = [xtarget, ytarget, ztarget]
                    if old_pos:
                        distance_from_old = np.linalg.norm(np.array(newpos) - np.array(old_pos))
                    else:
                        distance_from_old = 1000
                    if distance_from_old > 20:
                        arm.set_position(x=xtarget, y=ytarget, z=ztarget   , roll=177.9, pitch=1.1, yaw=87, speed=speed1, mvacc= mvacc1, wait=False, is_radian= False)
                        old_pos = newpos
                    time.sleep(0.1)
                    print(arm.get_position(), arm.get_position(is_radian=True))
                    # wait for keyboard input with python
                    #input("Press Enter to continue...")
                
                fo.write(",".join(map(str, three_d_point)) + "\n")
                
            frame = np.concatenate((frame1, frame2), axis=1)
            mask_comb = np.concatenate((mask1, mask2), axis=1)
            # add third dimension to mask with 3 channels
            mask_comb = np.repeat(mask_comb[:, :, np.newaxis], 3, axis=2)
            # combien frame and mask vertically
            frame = np.concatenate((frame, mask_comb), axis=0)
            cv2.imshow("frame", frame)
            # save the image
            cv2.imwrite(f"streams_img/{id}/{n}.jpg", frame)
            n += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("quit")
                break


def vis_3d(file="streams/1027.txt"):
    points = []
    with open(file) as f:
        for i, line in enumerate(f):
            
            points.append(list(map(float, line.split(","))))
    # display 3d plot, coloured by the time
    time = np.arange(0, len(points))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=time)
    plt.show()
    

if __name__ == '__main__':
  print("hi")
  fire.Fire(main)
print('hi')