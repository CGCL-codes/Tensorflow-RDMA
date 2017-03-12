bazel clean
./configure 
#bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
# To build with GPU support:
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# The name of the .whl file will depend on your platform.
pip install --upgrade /tmp/tensorflow_pkg/tensorflow-0.10.0-py2-none-any.whl
