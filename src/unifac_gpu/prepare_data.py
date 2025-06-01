from .utils import checkCudaErrors
from cuda.bindings import driver

def load_program():
    # Initialize CUDA Driver API
    checkCudaErrors(driver.cuInit(0))

    # Retrieve handle for device 0
    cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

    print(cuDevice)