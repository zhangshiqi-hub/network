class Config():
    def __init__(self, input_size, hidden_size, output_size,nhead,dropout,feature_dim=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nhead=nhead
        self.dropout =dropout
        self.feature_dim =feature_dim
        self.encoder_use_A_in_attn=True
        self.encoder_use_D_in_attn=False
        # decoder config
        self.n_decode_layers=6
        self.decoder_use_A_in_attn=True
        self.decoder_use_D_in_attn=True

