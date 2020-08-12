Say we have a network:

```python
alexnet = models.alexnet(pretrained=True)
```

If the layer name or size of layer is unknown, the following code is useful for determining the architecture of the network. Neural network models in PyTorch consist of [modules](https://pytorch.org/docs/stable/nn.html), which can be composed of other modules or layers of the network. 

```python
for module in enumerate(alexnet.named_children()):
    print(module)
```

outputs:
```
(0, ('features', Sequential(
  (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (4): ReLU(inplace=True)
  (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU(inplace=True)
  (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): ReLU(inplace=True)
  (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
)))
(1, ('avgpool', AdaptiveAvgPool2d(output_size=(6, 6))))
(2, ('classifier', Sequential(
  (0): Dropout(p=0.5, inplace=False)
  (1): Linear(in_features=9216, out_features=4096, bias=True)
  (2): ReLU(inplace=True)
  (3): Dropout(p=0.5, inplace=False)
  (4): Linear(in_features=4096, out_features=4096, bias=True)
  (5): ReLU(inplace=True)
  (6): Linear(in_features=4096, out_features=1000, bias=True)
)))
```

This reveals that the Alexnet model consists of three modules: *features*, *avgpool*, *classifier*, these are the names we can use to access the modules. During a forward pass, data will flow through each module in that order. Each module can consist of multiple layers, here we find that *features* has 13 layers, numbered 0 through 12, *avgpool* has just one layer, and *classifier* has 7 layers, numbered 0 through 6. The layers of a module are accessed similar to a list; for example, to access the last layer of the classifier module in our Alexnet:

```python
alexnet.classifier[-1]
```

Now that we know how to reference the layers of a network, we can register a forward hook to access to the layer's activations during a forward pass. The *register_forward_hook()* function takes as argument a function that is executed during a forward pass through that layer.

```python
alexnet.classifier[-1].register_forward_hook(my_hook_function())
```

During activation maximization, the forward hook into a layer allows the gradient ascent loop to access the activation of the target neuron and perform a backward pass from that neuron.