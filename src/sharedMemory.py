# if __name__ == "__main__":
#     pass

import mmap
import os
import sys
import time

FRAME_SIZE = 829440

def openSharedMemory():
    shmName = "shared_frame_1"
    shmFd = os.open(f"/dev/shm/{shmName}",os.O_RDWR)
    if shmFd < 0:
        raise OSError("Failed to create shared memory")

    os.ftruncate(shmFd, FRAME_SIZE)

    mapObj = mmap.mmap(shmFd, FRAME_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)

    if mapObj == -1:
        os.close(shmFd)
        raise OSError("Failed to map shared memory block")

    return mapObj
