from genie.attention import BasicSelfAttention, MemoryEfficientAttention
import torch
import onnxruntime as ort

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

output_path =  "/tmp/xformer_attn.onnx"
torch.onnx.export(net, dummy_input, output_path, verbose=True)
cpu = False
providers = ["CPUExecutionProvider"] if cpu else ["CUDAExecutionProvider"]
net = ort.InferenceSession(output_path, providers=providers)