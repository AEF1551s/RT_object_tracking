# if __name__ == "__main__":
#     pass

import numpy as np
import cv2 as cv

def YUV422toBGR(yuvData, width, height):
    yuvFrameRAW = np.frombuffer(yuvData, dtype=np.uint8, count = -1, offset = 0)
    yuvFrameRAW = yuvFrameRAW.reshape((height, width * 2)) 
    yuvFrameRAW = yuvFrameRAW.reshape((height, width, 2)) 
    bgrFrame = cv.cvtColor(yuvFrameRAW, cv.COLOR_YUV2BGR_Y422)
    return bgrFrame

def YUV422toRGB(yuvData, width, height):
    yuvFrameRAW = np.frombuffer(yuvData, dtype=np.uint8, count = -1, offset = 0)
    yuvFrameRAW = yuvFrameRAW.reshape((height, width * 2)) 
    yuvFrameRAW = yuvFrameRAW.reshape((height, width, 2)) 
    rgbFrame = cv.cvtColor(yuvFrameRAW, cv.COLOR_YUV2RGB_Y422)
    return rgbFrame
