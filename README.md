# Neural Network Cellular Automata and Open-ended Evolution
Differentiable neural network cellular automata.

Initially inspired by [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (A. Mordvintsev et al., 2020).

The goal of this project is to train neural cellular automata to generate an ever-growing set of diverse patterns and behaviours to replicate the properties of open-ended systems. 

### Notebooks
1. CA_totrain.ipynb - replicating the work from [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) in pytorch.
2. CA_totrain_embedder.ipynb - using a basic convnet embedder to try to train a CA network to make diverse patterns (not working atm.)
3. CA_pop_totrain_embedder.ipynb - same as above but now with a population of CA rules (seems to work, behaves better with the triplet loss)

