from setuptools import setup

setup(
    name="graphmix",
    version="0.1.0",
    author="Yining Shi",
    author_email="shiyining@pku.edu.cn",
    description="GraphMix : scalable minibatch server for GNN training",
    # ext_modules=[CMakeExtension("graphmix")],
    # cmdclass={"build_ext": CMakeBuild},
    packages=['graphmix', 'graphmix.dataset', 'graphmix.tensorflow', 'graphmix.torch'],
    package_dir = {'': 'python'},
    zip_safe=False,
)
