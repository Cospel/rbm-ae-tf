# rbm-ae-tf
Tensorflow implementation of Restricted Boltzman Machine and Autoencoder for layerwise pretraining of Deep Autoencoders with RBM. Idea is to first create RBMs for pretraining weights for autoencoder. Then weigts for autoencoder are loaded and autoencoder is trained again. In this implementation you can also use tied weights of autoencoder.

More about pretraining of weights in this paper:

##### https://www.cs.toronto.edu/~hinton/science.pdf

I was inspired with these implementations but I need to refactor them and improve them(thanks). I tried to use also similar api as it is in [tensorflow/models](https://github.com/tensorflow/models):
> [LINK1](https://www.snip2code.com/Snippet/1059693/RBM-procedure-using-tensorflow)
> [LINK2](https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5)


Feel free to make updates, repairs. You can enhance implementation with some tips from:
> [Practical Guide to training RBM](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
