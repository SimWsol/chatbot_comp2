import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(input_dim, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input_seq):
        embedded_seq = self.dropout_layer(self.embedding_layer(input_seq))
        rnn_outputs, hidden_state = self.rnn(embedded_seq)
        return rnn_outputs, hidden_state


class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.size()
        decoder_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attention_layer(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        scores = self.score_layer(energy).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf') if scores.dtype == torch.float32 else -1e4)

        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, attention, dropout_rate=0.1, pad_idx=0):
        super(Decoder, self).__init__()
        self.pad_idx = pad_idx
        self.output_dim = output_dim
        self.attention = attention
        self.embedding_layer = nn.Embedding(output_dim, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_token, decoder_hidden, encoder_outputs, source_seq=None, return_attention=False):
        input_token = input_token.unsqueeze(1)
        embedded_token = self.dropout_layer(self.embedding_layer(input_token))

        mask = (source_seq != self.pad_idx).int() if source_seq is not None else None
        attention_weights = self.attention(decoder_hidden, encoder_outputs, mask=mask).unsqueeze(1)

        context_vector = torch.bmm(attention_weights, encoder_outputs)
        rnn_input = torch.cat((embedded_token, context_vector), dim=2)

        rnn_output, hidden_state = self.rnn(rnn_input, decoder_hidden)
        predictions = self.fc_layer(torch.cat((rnn_output, context_vector), dim=2)).squeeze(1)

        if return_attention:
            return predictions, hidden_state, attention_weights
        else:
            return predictions, hidden_state


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = source_seq.size(0)
        target_len = target_seq.size(1)
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        encoder_outputs, hidden_state = self.encoder(source_seq)
        input_token = target_seq[:, 0]

        for t in range(1, target_len):
            output, hidden_state = self.decoder(input_token, hidden_state, encoder_outputs, source_seq)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = target_seq[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs


def beam_search(model, tokenizer, input_seq, beam_width=3, max_length=30):
    device = input_seq.device
    start_token = tokenizer.word2idx.get("<start>", 2)
    end_token = tokenizer.word2idx.get("<end>", 3)
    pad_token = tokenizer.word2idx.get("<pad>", 0)

    encoder_outputs, hidden_state = model.encoder(input_seq)
    beams = [(torch.tensor([start_token], device=device), hidden_state, 0.0)]

    for _ in range(max_length):
        candidates = []
        for seq, hidden, log_prob in beams:
            input_token = seq[-1].unsqueeze(0)
            output, new_hidden, _ = model.decoder(input_token, hidden, encoder_outputs, return_attention=True)

            probabilities = torch.softmax(output.squeeze(0), dim=-1)
            top_probs, top_indices = torch.topk(probabilities, beam_width)

            for i in range(beam_width):
                token = top_indices[i].item()
                token_log_prob = torch.log(top_probs[i] + 1e-10).item()
                new_seq = torch.cat([seq, torch.tensor([token], device=device)])
                candidates.append((new_seq, new_hidden, log_prob + token_log_prob))

        beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]

        if all(seq[-1].item() == end_token for seq, _, _ in beams):
            break

    best_seq, _, _ = beams[0]
    return tokenizer.decode([t.item() for t in best_seq if t.item() not in {start_token, end_token, pad_token}])