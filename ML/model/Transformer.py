import torch.nn as nn

# Transformer模型定义
class TransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, num_heads=8, num_layers=6, d_ff=512, dropout=0.1):
        super(TransformerNet, self).__init__()

        # Transformer 模型组件
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        # 编码器部分
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=num_heads, 
                dim_feedforward=d_ff, 
                dropout=dropout
            ), 
            num_layers=num_layers
        )

        # 输出层
        self.fc = nn.Linear(input_dim, output_dim)  # 输出一个连续值

    def forward(self, x):
        # Transformer 输入需要是 (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # 转换形状为 (seq_len, batch_size, input_dim)

        # 编码器输出
        x = self.encoder(x)
        
        # 只取最后一个时刻的输出作为回归特征
        x = x[-1, :, :]  # 选择序列中的最后一个位置 (可以根据需求修改)

        # 回归任务，不使用激活函数
        x = self.fc(x)
        return x