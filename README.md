# DiSMEC++
A software package for large-scale linear multilabel classification. `DiSMEC++` is a new implementation
of the [`DiSMEC`](https://github.com/xmc-aalto/dismec) algorithm that is better adapted many-core CPUs found in modern cluster environments by 
taking into account the NUMA setup of these systems, and provides a variety of additional features over 
the original implementation. Some highlights are:
* Ability to read data from and store models as `npy` files
* New weight initialization algorithms for faster training
* Ability to handle dense input features
* Additional loss functions and regularizers


## Building
The code can be build with cmake:
```shell script
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_DEVEL_DOCS=Off -S . -B ./build
cmake --build build --parallel
```
The resulting executables and python library will be placed in `./build/bin`
and `./build/lib` respectively. 

**Attention** Note that dependencies of the code are included in the `deps` directory as
git submodules. Please ensure that these are properly initialized before building, e.g., by
`clone`ing the repository with the `--recurive` argument.

## Usage
To train a model, the `train` program can be used, new predictions can be made
using the `predict` executable. Calling these programs with `--help` as argument
will display the list of command-line arguments with a short description.

### Example
Suppose you want to train a model for the <cite>Eurlex-4k</cite> [[1]](#1) dataset. 
You can download the data from the 
[extreme classification repository](http://manikvarma.org/downloads/XC/XMLRepository.html).
The two files `eurlex_train.txt` and `eurlex_test.txt` are assumed to be in the same directory
as the executables, otherwise you need to adjust the paths. 

A typical training call would look like this:
```shell
./train train_eurlex.txt eurlex.model --augment-for-bias --normalize-instances --save-sparse-txt --weight-culling=0.01
```
The first positional argument specifies the source for the training data, `train_eurlex.txt`
and the second argument the destination for the result, `eurlex.model`. Note that models are
distributed over several files, with `eurlex.model` serving as an index file, and the actual
weights saved in files that match `eurlex.models.weights*`.

The next two arguments specify the preprocessing to be applied to the data: 
`--augment-for-bias` adds a feature column of ones that serve as a bias term, and
`--normalize-instances` normalizes the features of each instance to have Euclidean
norm of 1, which usually improves the results. 

The last two arguments specify the saving of the model. The `--save-sparse-txt` indicates
that model weights are saved as human-readable text files of sparse matrices in a `index:value` 
format. To ensure model sparsity, `--weight-culling=0.01` removes all weights with a magnitude
of less than `0.01` from the saved model.

To test the model, you can run
```shell
./predict test_eurlex.txt eurlex.model eurlex_pred.txt --augment-for-bias --normalize-instances --topk=5
```
Note that the same preprocessing arguments as for the training run need to be specified. This
call will determine the top-5 labels for all instances in the test set and write them to the
`eurlex_pred.txt` file. It will also print out a multitude of metrics that can be calculated
from these predictions. If you want to further process these metrics, it might be helpful to
have them in a format that is easier for scripting. This can be achieved using the argument
`--save-metrics=eurlex-metrics.json`, which will write the metrics to the given json file that
can be imported, e.g., into a python script.


## Documentation
If you have doxygen installed, then you can build the documentation 
using `cmake --build build --target doxygen` and will find the documentation
in `build/docs/html`.

## Directories
### `src`
Contains the C++ code for the DiSMEC solver and the python bindings.

### `deps`
Contains sources/headers for libraries that we use as part of the implementation,
as git submodules. Currently, these are 
* https://github.com/CLIUtils/CLI11
* https://github.com/gabime/spdlog
* https://github.com/onqtam/doctest
* https://github.com/pybind/pybind11
* https://github.com/nlohmann/json
* https://gitlab.com/libeigen/eigen

Additionally, we expect [Boost](https://www.boost.org/) to be available on the system.

## References
<a id="1">[1]</a>
E. L. Mencia, and J. Furnkranz, Efficient pairwise multilabel classification for large-scale problems in the legal domain in ECML/PKDD, 2008. 

I. Chalkidis, E. Fergadiotis, P. Malakasiotis, N. Aletras and I. Androutsopoulos, Extreme Multi-Label Legal Text Classification: A Case Study in EU Legislation, in Natural Legal Language Processing Workshop 2019.  
