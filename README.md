# Some thoughts about dropout with ReLU activation

There are several ways to implement dropout in a neural network. 
Ordinary dropout like most described used a form to drop the neurons in the feed forward process.

It could look like this:

```
            float dropout = 0.5f;
            for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1]) // layer
                for (int k = 0, kEnd = u[i + 1], nEnd = t + u[i]; k < kEnd; k++, j++) // neurons
                {
                    float net = 0;
                    for (int n = t, m = w + k; n < nEnd; n++, m += kEnd) // weights                
                        net += neuron[n] * weight[m];

                    if (i == layer - 1) // output layer prepare for softmax
                        neuron[j] = net; 
                    if (FastRand() / 32767.0 < dropout)
                        neuron[j] = 0; // dropout
                    else
                        neuron[j] = net > 0 ? net / (1 - dropout) : 0; // relu activation
                }//--- k ends  
```

Dropout comes in after the netinput is computed, no matter if the neuron was activated or not.
But in combination with the ReLU activation, the ReLU themself represents a kind of dropout.
So here we must expect heavy problems if we dont carefully implement dropout.

Another approach is inverted dropout
https://jamesmccaffrey.wordpress.com/2019/05/07/neural-network-dropout-and-inverted-dropout/


Lets see the high level picture in pseudo code how this could work
```
for n data
    FF()   
    
    // inverted dropout
    if (isTraining && dropout > 0)
                for (int i = input; i < inputHidden; i++)
                {
                    // inverted single dropout
                    if (FastRand() / 32767.0 < dropout)
                        neuron[i] = -1; // -1 for a dropout neuron, thx to relu ^^
                    else
                        neuron[i] /= 1 - dropout;
                }
    BP()
    Optimizer()
```

So if I understand the technique right, it is a much better dropout, because the information of the prediction in FF will be keeped.
The effect of inverted dropout comes in in the backpropagation only and make it for me much harder to understand the complete process.
And lastly if there are 2 activated neurons only left and they drop, the network cannot predict anymore, no matter which dropout is used.

My advantage is goodgame, with that tool I can see how the neural network works.
It could look like this.

And in a good moment it seems I have found a better way to deal with dropout in combination with ReLU.

The key is to track the activation level on each layer, let's assume the hidden layer1 has 16 neurons and 4 of them are activated.
Then we take this probability like so 4 / 16 = 0.25 = currentDrop and the dropout = 0.5 and just multiply them 0.25 * 0.5 = 0.125 to a more adaptive dropout rate.
Now let's assume we have only one activated neuron left 1 / 16 = 0.0625, multiplied with dropout 0.0625 * 0.5 = 0.03125 it comes to a really small probability that dropout comes in. 
In contrast if all neurons were activated 16 / 16 = 1, dropout would work over all neurons and the droutout rate with 0.5 would fully comes in. 
Cool stuff.

It's not a gurantee that the network works fine in every situation, but with that technique it is a huge step to that goal.


```
            for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1]) // layer
            {
                int reluActivationCnt = 0;
                for (int k = 0, kEnd = u[i + 1], nEnd = t + u[i]; k < kEnd; k++, j++) // neurons
                {
                    float net = 0;
                    for (int n = t, m = w + k; n < nEnd; n++, m += kEnd) // weights                
                        net += neuron[n] * weight[m];

                    if (i == layer - 1) // output layer prepare for softmax
                        neuron[j] = net; 
                    if (FastRand() / 32767.0 < dropout)
                        neuron[j] = 0; // dropout
                    else if(net > 0) // relu
                    {
                        reluActivationCnt++;
                        neuron[j] = net; // relu activation
                    }
                    else
                        neuron[j] = 0; 
                }//--- k ends  

                if (dropout > 0) // if dropout is used
                    if (i != layer - 1) // if hidden neuron
                    {
                        float relu_prop = reluCnt / (float)u[i + 1];
                        int jj = j - u[i + 1];
                        for (int k = 0, kEnd = u[i + 1], nEnd = t + u[i]; k < kEnd; k++, jj++) // neurons on this layer
                            if (FastRand() / 32767.0 < dropout * relu_prop)
                                neuron[jj] = -1; 
                            else
                                neuron[jj] /= ((relu_prop + (1 - dropout)) / 2.0f);
                    }
            }
```






