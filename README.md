# GraphMix
GraphMix is a graph sampling library designed for distributed GCN training.
## Installation
We have provided a conda environment file .

```shell
conda env create
conda activate graphmix
```

If you don't use a conda environment. You may install metis, protobuf, zeromq, cmake via apt and install pybind11 via pip.

Ather these requirements are installed, build with cmake.

```shell
mkdir build
cd build
cmake ..
make -j8
```

After building, set the PYTHONPATH environment via

```shell
./env.sh
```

## Usage

To launch distributed GCN, there will be two steps.

1. Prepare graph data with graph partition
2. Prepare a launch script

### Prepare Graph partition

You can partition a graph into several parts via the following commands

```
python3 graphmix.partition -d Reddit -n 4 -p ~/mydata
```

This command creates a partitioned graph using metis partition with 4 parts under ~/mydata.

We have several dataset prepared like Reddit, Yelp, Flickr, ogbn-arxiv, ogbn-products, Cora, PubMed.

You will find a meta.yml file are some parts directory.

### Prepare Launch script

A minimal launch script is like this:

```yaml
# config.yml
env:
  GRAPHMIX_ROOT_URI : 123.123.1.2
  GRAPHMIX_ROOT_PORT : 8889
  GRAPHMIX_NUM_WORKER : 4
launch:
  worker : 4
  server : 4
  scheduler : true
  data : ~/mydata/Reddit
```

The three environment variables defines the the root address so that the communication could setup.

launch defines how many processes are launched by this script. Currently if you want to run on multiple machines, you have to write a launch script on each machine.

There are several rules about launch process.

1. The sum of server must be equal to graph partition number.
2. There should be only one scheduler and it should be launched at the machine holding the GRAPHMIX_ROOT_URI.
3. data should point to the data path created by graph partition.
4. Graph server are indexed by ip, from 0 to n-1. The corresponding partition file should exist under data path.

### Run

When launch script is prepared. A minimal python script to use

```python
# example.py
import argparse
import graphmix

def test(args):
    comm = graphmix.Client()
    print(comm.meta)
    query = comm.pull_graph()
    graph = comm.wait(query)
    print(graph)

def server_init(server):
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=128, width=2, depth=2)
    server.is_ready()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    graphmix.launcher(test, args, server_init=server_init)

```

by running example.py with a config script in config.

```shell
python3 example.py config.yml
```

In this example, the server first create a GraphSage sampler.  The worker create an async query to pull a minibatch and use wait to wait for the minibatch to be ready.
