from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="w4a16",
    version="0.1.0",
    packages=["w4a16"],
    ext_modules=[
        CUDAExtension(
            name="w4a16._C",
            sources=[
                "csrc/int4_linear.cpp",
                "csrc/int4_linear_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-gencode", "arch=compute_75,code=sm_75"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
