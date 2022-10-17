import cv2
import time
import os

def check_clear_buffer(clear_buffer : bool, buffer_counter : int):
    if clear_buffer:
        buffer_counter += 1
        print("clearing..." + str(buffer_counter))
        if buffer_counter == 100:
            print("clearing done")
            buffer_counter = 0
            return False, buffer_counter
    return clear_buffer, buffer_counter


def create_images(port):
    cap = cv2.VideoCapture(port)
    if port == 2:
        camera = "right"
    elif port == 6:
        camera = "left"
    else:
        raise ValueError()

    print("#### Create images for {} camera ####".format(camera))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = (6, 4)
    index = 0
    buffer_counter = 0
    clear_buffer = False

    while (True):
        _, frame = cap.read()
        ret, corners = cv2.findChessboardCorners(frame, chessboard_size, None)
        cv2.imshow("Video", frame)
        outputR = frame.copy()

        clear_buffer, buffer_counter = check_clear_buffer(clear_buffer, buffer_counter)

        if ret and not clear_buffer:
            cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(outputR, chessboard_size, corners, ret)
            cv2.imshow('Corners', outputR)
            cv2.imwrite("{}/runs/1/{}_{}.png".format(os.path.dirname(__file__), camera, index), frame)
            index += 1
            clear_buffer = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    create_images(2)
    create_images(6)

