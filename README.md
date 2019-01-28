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
          -  outputs PILImage images of range[0,1] dataloader: in **torch.utils.data.DataLoader()**

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
