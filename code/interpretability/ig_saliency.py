def integrated_gradients(inputs, model, baseline=None, steps=20, gpu_device="cuda:0"):
    """
    inputs: One numpy array of a figure, size of which fit the input of the network
    model: The network
    steps: The steps that use sum to estimate integration
    gpu_device: Use gpu or cpu
    """
    # Set the baseline if not exist
    if baseline is None:
        baseline = 0 * inputs
    # Slowly change the figure and saves in an array
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)]
    # Set an array to save gradients
    gradients = []
    # It is a problem that we must set train here.
    model.eval()
    for input_np in scaled_inputs:
        # Change the input to tensor
        input_tensor = torch.tensor(input_np, requires_grad=True, dtype=torch.float32, device=gpu_device)
        # Calculate the spike train
        output_raw = model(input_tensor)

        # Define the value we are interested in
        val = torch.sum(output_raw)

        # Reset the gradient and status of the network
        model.zero_grad()
        # Use backward to calculate the gradient
        val.backward()
        # Detach the gradient and save
        gradient = input_tensor.grad.detach().cpu().numpy()
        gradients.append(gradient)
    # Calculate the integrated gradients by multiply the average gradients and value difference
    avg_grads = np.average(gradients, axis=0)
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad
