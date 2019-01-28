# pytorch_tutorial
This is a Pytorch tutorial and personal notes, comments

1. 60 minute Blitz
    1. manipulating tensors
    Nothing spectacular here.
    NumPy-like:
      * indexing works
      * resizing: x.view(-1) # the size -1 is inferred from other dimensions
      * x.item() returns the value of the tensor if it's a scalar
      * Tensor to NumPy: x.numpy()
      * NumPy to Tensor: torch.from_numpy(x)

   CUDA Tensors:
      * find enabled CUDA devices: torch.cuda_is_available("cuda")
      * CUDA  device object: torch.device("cuda")
      * create a tensor on GPU: torch.ones_like(x, device=device)
      * move a tensor to GPU: x.to(device)
 
