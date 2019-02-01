# pytorch_tutorial
This is a Pytorch tutorial and personal notes, comments

1. ### 60 minute Blitz
    1. #### manipulating tensors
    Nothing spectacular here.

    **NumPy-like**:
      * indexing works
      * resizing: x.view(-1)
      The size -1 is inferred from other dimensions
      * x.item() returns the value of the tensor if it's a scalar
      * Tensor to NumPy: x.numpy()
      * NumPy to Tensor: torch.from_numpy(x)

    **CUDA Tensors**:
      * find enabled CUDA devices: torch.cuda_is_available("cuda")
      * CUDA  device object: torch.device("cuda")
      * create a tensor on GPU: torch.ones_like(x, device=device)
      * move a tensor to GPU: x.to(device)

    2. ### Autograd

    **Tensor**:
      * track ops: x = torch.ones(2,2,**requires_grad**=True)
      By default, it is set to False
      * tensor as result of op:  it has a **grad_fn**

    **Gradients**:
      * if requires_grad is set to True, the vector has a backward method and a grad attribute
      * with torch.no_grad(): forces whatever is there not to use grad

    3. ###  Neural Networks
      packages: **torch.nn** and **torch.nn.functional**
      * defining a network:
        - class Net(nn.Module)
        - super(Net, self).__init__
        - only need to define the forward pass, **autograd** automatically creates the backward pass
        - net.parameters : list of parameters, each is a set of weights to be tuned
        - **net.zero_grad()**: set all gradients to zero, reset
      torch.nn only support mini-batches. In case of single sample, use **input.unsqueeze(0)** to fake a batch dimension.

      * Loss function
      All loss functions are defined in **torch.nn**.
      Calling **loss.backward()**, the whole graph is differentiate with respect to the loss.

      * Backpropagation
      We only need to do **loss.backward()**. We need to clear the existing gradients, otherwise gradients will be accumulated to existing gradients.

      * Updating weights
      packages: **torch.optim as optim**
      5 steps:
        - declare optimizer
        - optimizer.zero_grad()
        - loss = criterion(output, target)
        - loss.backward()
        - optimizer.step()

      4. ###  Training a classifier

      * **torchvision**:
          -  outputs PILImage images of range[0,1]
          -  dataloader: in **torch.utils.data.DataLoader()**

      * **training procedure**:
         - get the inputs
         - reset the optimizer
         - forward + backward + optimize
         - print statistics

      * **testing procedure**:
         - force no_grad with **with torch.no_grad()**
         - no backward
         - correct+= (predicted == labels).sum().item()

      * **Using GPU**:
         - define our GPU device as 'cuda:0': torch.device("cuda:0" if **torch.cuda.is_available()** else "cpu")
         - migrate everything to GPU: net.to(device)
         - we will have to seed the inputs and targets at every step

2. ### Text
    1. #### Chatbot tutorial

    * **preprocessing text**

    This is a fundamental step in DL, especially in text-related tasks.
    We need to understand the structure of the base file we are given.
    Many steps are necessary:
      - create formatted data file: here we want to tidy up the base file
      so that it is easily readable.
      - create a vocabulary: mapping each unique word to an index value, and
      an inverse mapping of indexes to words
      - convert Unicode strings to ASCII, convert to lowercase, trim all non
      characters
      - trim rarely used words: soften the difficulty


      * **Prepare data for models**

      We need to convert our data into torch tensors. We must be aware of the
      variation of sentence length in our batches. We can use zero-padding for
      shorter sentences.
      Batch size is now (batch_size, max_length).
      We transpose so that indexing across the first dimension returns a time
      step across all sentences in the batch.
      Good batch size is therefore (max_length, batch_size)

      * **Define Models**

      Here are some elementary elements of the encoder model:
      - GRU: Gated Recurrent Unit. In order to encode both past and future
        context, we can use a bidirectional GRU.
      - pack and unpack pading

      Here are some elementary elements of the decoder model:
      - generate response sentence in a token-by-token fashion
      - attention mechanism: allows the decoder to pay attention to certain
        parts of the input sequence.
      - score functions: ways of calculate the attention energies between the
        encoder output and decoder output.
      - Attention layer


      * **Define Training Procedure**

      Keys ideas:
      - masked loss: when calculating the loss, we have to take into account
        the padding.
      - teacher forcing: at some probability, we use the current target word
        as the decoder's next input rather than using the decoder's current
        guess. It can lead to model instability during inference
      - gradient clipping: we prevent the gradients from growing exponentially
        and either overflow or overshoot steep cliffs in the cost function.


      * **Define Evaluation**

      We have to get rid of teacher forcing: we simply choose
      the word from decoder_output with the highest softmax value.
