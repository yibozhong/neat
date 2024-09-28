import torch
from torch import nn
import inspect
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import torch.nn.functional as F

class neat_vanilla(nn.Module):
    """
    vanilla implementation of Neat, no residual, no depth
    use ReLU as the default non-linear function
    args:
        dim: hidden dimension (a.k.a. rank)
        out_dim: output dimension
    """
    def __init__(self, dim=32, out_dim=768):
        super().__init__()
        self.non_linear = nn.ReLU()
        self.A = nn.Linear(out_dim, dim, bias=False)
        self.B = nn.Linear(dim, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x, weight):
        delta_w = self.non_linear(weight @ self.A.weight.t()) # non-linear(W0 A)
        x = x @ delta_w
        x = self.B(x)
        return x

def forward_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states) + (self.q_adapter(hidden_states, self.query.weight.t()) * self.s)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states) + (self.v_adapter(encoder_hidden_states, self.value.weight.t()) * self.s))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states) + (self.v_adapter(hidden_states, self.value.weight.t()) * self.s))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states) + (self.v_adapter(hidden_states, self.value.weight.t()) * self.s))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def set_neat_s(model, s=1.0):
    """
    small version of Neat, only adapting layer 4 and onward, with rank = 1
    """
    for n, layer in model.named_modules():
        if n.endswith(('self')) and ('10' in n or '11' in n or '8' in n or '9' in n or '6' in n or '7' in n or '5' in n or '4' in n):
            layer.q_adapter = neat_vanilla(dim=1)
            layer.v_adapter = neat_vanilla(dim=1)
            bound_method = forward_attn.__get__(layer, layer.__class__)
            layer.s = s
            setattr(layer, 'forward', bound_method)

def set_neat_l(model, dim=8, s=1.0):
    """
    large version of Neat, adapting all the layers
    default rank is 8 to but it can be flexibly set
    """
    for n, layer in model.named_modules():
        if n.endswith(('self')):
            layer.q_adapter = neat_vanilla(dim=dim)
            layer.v_adapter = neat_vanilla(dim=dim)
            bound_method = forward_attn.__get__(layer, layer.__class__)
            layer.s = s
            setattr(layer, 'forward', bound_method)