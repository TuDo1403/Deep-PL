# from libs.utils import Encoder, Decoder
# import tensorflow as tf

# class Transformer(tf.keras.Model):
#   def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
#     super(Transformer, self).__init__()

#     self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

#     self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

#     self.final_layer = tf.keras.layers.Dense(target_vocab_size)

#   def forward(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

#     enc_output = self.tokenizer(inp, training, enc_padding_mask)

#     dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

#     final_output = self.final_layer(dec_output)

#     return final_output, attention_weights