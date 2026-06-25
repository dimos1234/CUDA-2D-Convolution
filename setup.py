from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_convolution',
    ext_modules=[
        CUDAExtension(
            name='custom_convolution',
            sources=['convolution_wrapper.cpp', 'conv_unrolled.cu'],
            define_macros=[('BUILDING_PYTHON_MODULE', None)] # This hides the main() function!
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
