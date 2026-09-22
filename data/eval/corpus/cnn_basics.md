# Convolutional Neural Networks

A convolutional neural network (CNN) is a neural network designed for grid-like data such as images. Instead of connecting every input pixel to every neuron, a convolution layer slides a small learned filter (kernel) across the input and produces a feature map. Because the same filter is reused at every position, CNNs have far fewer parameters than fully connected networks and they detect a pattern regardless of where it appears.

Pooling layers reduce the spatial size of feature maps. Max pooling keeps the largest activation in each window, while average pooling keeps the mean. Pooling lowers computation, enlarges the receptive field of later layers and gives a small amount of translation invariance.

A classic image classifier stacks convolution, ReLU activation and pooling blocks, then flattens the result and applies fully connected layers that output class scores. Batch normalisation stabilises training by normalising activations within a mini-batch, and dropout randomly disables units during training to reduce overfitting.

Well-known CNN architectures include LeNet-5 (1998, handwritten digits), AlexNet (2012, ImageNet), VGG (deep stacks of 3x3 convolutions) and ResNet, which introduced skip connections so that very deep networks remain trainable.
