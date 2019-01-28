import bin.module.util as util


TOKEN = util.general.SettingContainer(
    START = '_startTk',
    END = '_endTk'
)
MODEL = util.general.SettingContainer(
    INPUTS_ENCODER = 'inputs_encoder',
    INPUTS_DECODER = 'inputs_decoder',
    EMB_ENCODER = 'Emb_Encoder',
    EMB_DECODER = 'Emb_Decoder',
    LSTM_ENCODER = 'LSTM_Encoder',
    LSTM_DECODER = 'LSTM_Decoder',
    OUTPUTS = 'outputs'
)