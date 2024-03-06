# Overview
This directory contains code necessary to run the GraphSage algorithm. 
GraphSage can be viewed as a stochastic generalization of graph convolutions, 
and it is especially useful for massive, dynamic graphs that contain rich feature information. 

To run GraphSAGE, it needs to train on an example graph or set of graphs. 
After training, GraphSAGE can be used to generate node embeddings for previously unseen nodes or entirely new input graphs, 
as long as these graphs have the same attribute schema as the training data.

# Quick Start
You can follow the steps below to quickly get up and running with graphsage models. 
These steps will let you run quick inference locally.
```
python sample.py
```


# Requirements
```
cudatoolkit      11.2.2
cudnn            8.1.0.77
tensorflow-gpu   2.6.0
```

