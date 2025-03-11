import torch
from torch.utils.tensorboard import SummaryWriter
from models.vision_transformer import ViT

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(
            img_size=128,
            patch_size=4,
            num_frames=3,
            tubelet_size=1,
            in_chans=2,
            encoder_embed_dim=192,
            encoder_depth=1,
            encoder_num_heads=8,
            decoder_embed_dim=192,
            decoder_depth=1,
            decoder_num_heads=8,
            mlp_ratio=4,
            num_out_frames=1,
            patch_recovery='linear',
            checkpointing=False).to(device)

print(model)

summary(model, input_size=(2, 3, 128, 128), device=str(device))

dummy_input = torch.randn(1, 2, 3, 128, 128).to(device)  
torch.onnx.export(model, dummy_input, "ViT.onnx", opset_version=14)

writer = SummaryWriter(log_dir="./logs/model-1")
writer.add_graph(model, dummy_input)
writer.close()

