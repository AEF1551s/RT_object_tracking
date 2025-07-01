from collections import deque
import asyncio
import posix
import numpy as np
from .sharedMemory import openSharedMemory
from .imageConversions import YUV422toBGR, YUV422toRGB

class FrameCapture:
    def __init__(self, height = 576, width = 720, frameCount = 60, tick =0.02, stream = True):
        self.savedRGBFrames = deque(maxlen = frameCount)
        self.height = height
        self.width = width
        self.bgrFrame = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        self.rgbFrame = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        self.stream = stream
        self.tick = tick

        try:
            self.sharedMemory = openSharedMemory()
            print("Shared memory opened successfully!")
        except OSError as e:
            print(f"Error: {e}")

    def captureRawFrame(self):

        self.yuvData = self.sharedMemory

        self.bgrFrame = YUV422toBGR(self.yuvData, self.width, self.height)
        self.rgbFrame = YUV422toRGB(self.yuvData, self.width, self.height)

    async def captureRawStream(self):
        while self.stream:
            self.captureRawFrame()
            await asyncio.sleep(self.tick)

    def saveVideoWindow(self):
        self.savedRGBFrames.append(self.rgbFrame)

    def getBGRFrame(self):
        return self.bgrFrame

    def getRGBFrame(self):
        return self.rgbFrame

    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width
