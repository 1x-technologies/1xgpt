from genie.attention import BasicSelfAttention, MemoryEfficientAttention
import torch

net = BasicSelfAttention(num_heads=4, d_model=32)
net2 = MemoryEfficientAttention(num_heads=4, d_model=32)
# tie the weights
net2.load_state_dict(net.state_dict())
net.eval().to("cuda")
net2.eval().to("cuda")

dummy_input = torch.randn(1, 16, 32, dtype=torch.float32).to("cuda")
y1 = net(dummy_input, causal=True)
y2 = net2(dummy_input, causal=True)
print('max numerical difference', torch.abs(y1 - y2).max())
assert torch.allclose(y1, y2, atol=1e-06)

torch.onnx.export(net, dummy_input, "/tmp/xformer_attn.onnx", verbose=True)
