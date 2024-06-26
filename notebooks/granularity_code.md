
# Granularity Measurement
The code is pretty complex since we need to split every 1d tensor to 45 segments (since we have 45 layers), and do the rotation in each layer. The code has already been optimized to most efficient style instead of using loops.

## Single index rotation:
Do the same rotation on each layer for all events. Shift_index can be manually configured. In caloqvae dataset, difference between GT and Recon data goes the maximum when $\text{shift\_index} = 9 \times i $ since there are 9 voxels in radial direction. A shift of 9 will help to catch the difference between adjacent voxels in angular direction.
```python
def measure_granularity(data_tensor, shift_index = 0):
    segment_length = 144  # Length of each segment
    num_segments = 45     # Number of segments per sample
    num_samples = data_tensor.size(0)  # Number of samples in the data tensor

    shifts = torch.full((num_samples,), shift_index, dtype=torch.int64, device=data_tensor.device).unsqueeze(-1).expand(-1, num_segments)

    # The result is a tensor of shape (num_samples, num_segments, segment_length)
    segments = data_tensor.unfold(1, segment_length, segment_length)

    # Create an indices tensor of shape (num_samples, num_segments, segment_length)
    indices = torch.arange(segment_length, device=data_tensor.device).repeat(num_samples, num_segments, 1)
    # Adjust the indices by adding the shifts and applying modulo operation to wrap around
    indices = (indices + shifts.unsqueeze(-1)) % segment_length  # Ensure correct broadcasting

    # Gather elements from the segments tensor using the adjusted indices
    rotated_segments = torch.gather(segments, 2, indices)

    # Reshape the rotated_segments tensor back to the original shape of data_tensor
    result_tensor = rotated_segments.view(num_samples, -1)

    # Compute the difference between the original data tensor and the result tensor
    diffs = data_tensor - result_tensor

    return diffs
```

## Random rotation:
Now let's only focus on the most sensitive shifting indexes $9 \times i$. Do the same rotation on each layer, but using random $i\in[1, 15]$ on different events since there are 16 voxels in angular direction.
```python
def measure_stochastic_granularity(data_tensor):
    segment_length = 144  # Length of each segment
    num_segments = 45     # Number of segments per sample
    num_samples = data_tensor.size(0)  # Number of samples in the data tensor

    # Use PyTorch to generate a random integer array of shape (num_samples,) with values between 0 and 15
    random_array = torch.randint(0, 16, (num_samples,), dtype=torch.int64, device=data_tensor.device)

    # Multiply random_array by 9 and expand it to shape (num_samples, num_segments)
    shifts = (random_array * 9).unsqueeze(-1).expand(-1, num_segments)

    # Unfold the data tensor to create segments of length 144
    # The result is a tensor of shape (num_samples, num_segments, segment_length)
    segments = data_tensor.unfold(1, segment_length, segment_length)

    # Create an indices tensor of shape (num_samples, num_segments, segment_length)
    indices = torch.arange(segment_length, device=data_tensor.device).repeat(num_samples, num_segments, 1)
    # Adjust the indices by adding the shifts and applying modulo operation to wrap around
    indices = (indices + shifts.unsqueeze(-1)) % segment_length  # Ensure correct broadcasting

    # Gather elements from the segments tensor using the adjusted indices
    rotated_segments = torch.gather(segments, 2, indices)

    # Reshape the rotated_segments tensor back to the original shape of data_tensor
    result_tensor = rotated_segments.view(num_samples, -1)

    # Compute the difference between the original data tensor and the result tensor
    diffs = data_tensor - result_tensor

    return diffs
```

# Masking noise to the output events:
Here, $p$ is the probability that a voxel will get masked with a random noise and $coe$ is the intensity of the noise. The noise is a product of a uniform distribution and a normal distribution with 0 expectation value.
```python
class Final_Dropout(nn.Module):
    def __init__(self, p=0.5, coe = 1):
        super().__init__()
        self.p = p 
        self.coe = coe
    def forward(self, x):
        uni_noise = (torch.rand(x.size(), dtype=x.dtype, device=x.device)).float()
        mask = torch.bernoulli(torch.full_like(x, self.p, dtype=torch.float32))
        gaussian_noise = torch.randn_like(x)
        noise = uni_noise.float() * gaussian_noise * self.coe 
        masked_result = x + torch.sqrt(torch.abs(x)) * mask * noise
        rulu = nn.ReLU()
        masked_result = rulu(masked_result)
        return masked_result
```