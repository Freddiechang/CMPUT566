import torch
import numpy as np
def integrated_gradients(inputs, model, baseline=None, steps=20, threshold=0.5, gpu_device="cuda:0"):
    """
    inputs: One numpy array of a figure, size of which fit the input of the network
    model: The network
    steps: The steps that use sum to estimate integration
    gpu_device: Use gpu or cpu
    """
    # move model to device
    model.to(gpu_device)
    # Set the baseline if not exist
    if baseline is None:
        baseline = 0 * inputs
    # Slowly change the figure and saves in an array
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)]
    # Set an array to save gradients
    gradients = []
    outputs = []
    model.eval()
    for input_np in scaled_inputs:
        # Change the input to tensor
        input_tensor = torch.tensor(input_np, requires_grad=True, dtype=torch.float32, device=gpu_device)
        # Calculate the result
        output_raw = model(input_tensor)

        # Define the value we are interested in
        output_select = output_raw > threshold
        val = torch.sum(output_raw * output_select)

        # Reset the gradient and status of the network
        model.zero_grad()
        # Use backward to calculate the gradient
        val.backward()
        # Detach the gradient and save
        gradient = input_tensor.grad.detach().cpu().numpy()
        output = output_raw.detach().cpu().numpy()
        gradients.append(gradient)
        outputs.append(output)
    # Calculate the integrated gradients by multiply the average gradients and value difference
    avg_grads = np.average(gradients, axis=0)
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad, outputs
