import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimentions is a tuple of integers.
    Hint: use torch.ones and multiply by val, or use torch.zeros and add val.
    e.g. if dimensions = (2, 3), and val = 3, then the returned tensor should be of shape (2, 3)
    specifically, it should be:
    tensor([[3., 3., 3.], [3., 3., 3.]])
    """
    # Using torch.ones and multiplying by val
    res = torch.ones(dimensions) * val
    
    # Alternatively, using torch.zeros and adding val
    # res = torch.zeros(dimensions) + val
    return res

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    # Calculate the element-wise product of A and B
    res = A * B
    # Alternatively, you could use torch.mul(A, B)
    # res = torch.mul(A, B)
    return res 


def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i})
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for 
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.
         
    """
    # Calculate the matrix product of X and the transpose of W
    res = torch.matmul(X, W.T)
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i}) and add the bias.
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.
       """
     # Calculate the matrix product of X and the transpose of W
    product = torch.matmul(X, W.T)
    # Add the bias to the product
    res = product + b
    return res

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    Hint: use PyTorch `heaviside` function.
    """
    # Calculate the step function activation using torch.heaviside
    # res = torch.heaviside(sum_total, torch.tensor(1.0))
    res = torch.heaviside(sum_total, torch.tensor(0.0))
    return res

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    Hint: use the functions you implemented above.
    """
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    res = calculate_activation(sum_total)
    # res = None  # TODO: implement this function 
    return res