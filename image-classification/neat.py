import torch
from torch import nn
import inspect
import torch.nn.functional as F

class neat_vanilla(nn.Module):
    """
    vanilla implementation of Neat with no intermediate layers, no residual, depth = 2 (A and B)
    use ReLU as the default non-linear activation
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
        delta_w = self.non_linear(weight @ self.A.weight.t()) # non_linear(W0 A)
        x = x @ delta_w
        x = self.B(x)
        return x

class neat_multilayers_depth_6(nn.Module):
    """
    Simpler naive implementation of Neat with multilayers.
    We actually use this version in the paper for the analysis, and it DOES seem to perform better.
    However, in order to easily stack up multiple layers flexibly, a more general implementation is provided (see neat_multilayers below).
    """
    def __init__(self, dim=32, out_dim=768):
        super().__init__()
        self.non_linear = nn.ReLU()
        self.A = nn.Linear(out_dim, dim, bias=False)
        self.i1 = nn.Linear(dim, dim, bias=False)
        self.i2 = nn.Linear(dim, dim, bias=False)
        self.i3 = nn.Linear(dim, dim, bias=False)      
        self.i4 = nn.Linear(dim, dim, bias=False) # 4 intermediate layers      
        self.B = nn.Linear(dim, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x, weight):
        delta_w = self.non_linear(weight @ self.A.weight.t())
        residual = delta_w.clone() 
        delta_w = self.non_linear(self.i1(delta_w))
        residual_ = delta_w.clone()
        delta_w = self.non_linear(self.i2(delta_w))
        delta_w = self.non_linear(self.i3(delta_w))
        delta_w = delta_w + residual_ 
        delta_w = self.non_linear(self.i4(delta_w))
        delta_w = delta_w + residual
        delta_w = self.B(delta_w)
        return x @ delta_w

class neat_multilayers(nn.Module):
    """
    Implementation of Neat with residual and intermediate layers
    use ReLU as the default non-linear activation
    args:
        dim: hidden dimension (a.k.a. rank)
        out_dim: output dimension
        depth: number of adapters in Neat (A, B and all the intermediate layers)
    """
    def __init__(self, dim=32, out_dim=768, depth=4):
        super().__init__()
        self.non_linear = nn.ReLU()
        self.depth = depth
        self.res_num = (depth - 2) // 2
        self.A = nn.Linear(out_dim, dim, bias=False)
        num_of_intermediate_layers = self.depth - 2
        assert (num_of_intermediate_layers % 2 == 0) # make sure residual works
        self.encoders = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_of_intermediate_layers // 2)])
        self.decoders = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_of_intermediate_layers // 2)])
        self.B = nn.Linear(dim, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x, weight):
        delta_w = self.non_linear(weight @ self.A.weight.t())  # (W_0 W_1) [a, d]
        residuals = []
        for i in range(self.res_num):
            residuals.append(delta_w.clone())
            delta_w = self.non_linear(self.encoders[i](delta_w))
        for i in range(self.res_num):
            delta_w = self.non_linear(self.decoders[i](delta_w))
            delta_w = delta_w + residuals[self.res_num - 1 - i]
        delta_w = self.B(delta_w)
        return x @ delta_w
    
def forward_attn(
        self, hidden_states, head_mask = None, output_attentions: bool = False
        ):
    mixed_query_layer = self.query(hidden_states) + 1.0 * self.q_adapter(hidden_states, self.query.weight.t())

    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states) + 1.0 * self.v_adapter(hidden_states, self.value.weight.t()))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        head_mask,
        self.attention_probs_dropout_prob if self.training else 0.0,
        is_causal=False,
        scale=None,
    )

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    return context_layer, None

def forward_ffn(self, input):
        return F.linear(input, self.weight, self.bias) + 1.0 * self.adapter(input, self.weight.t())
 
def set_neat_vanilla(model, neat_mode=1, mhsa_dim=16, ffn_dim=16):
    """
    neat mode:
        1: vanilla neat on mhsa q, v
        2: vanilla neat on both mhsa q, v and ffn
    """
    for name, layer in model.named_children():
        if 'attention' in name:
            layer.attention.neat_mode = neat_mode
            layer.attention.q_adapter = neat_vanilla(dim=mhsa_dim)
            layer.attention.v_adapter = neat_vanilla(dim=mhsa_dim)
            bound_method = forward_attn.__get__(layer.attention, layer.attention.__class__)
            setattr(layer.attention, 'forward', bound_method)
        elif 'dense' in name:
            if neat_mode == 1:
                continue
            layer.neat_mode = neat_mode
            layer.adapter = neat_vanilla(dim=ffn_dim, out_dim=layer.weight.shape[0])
            bound_method = forward_ffn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_neat_vanilla(layer, neat_mode, mhsa_dim, ffn_dim)


def set_neat_multilayers(model, neat_mode=1, mhsa_dim=16, ffn_dim=16, depth=2):
    """
        neat mode:
        1: neat with more depth on mhsa q, v (qv setting)
        2: neat with more depth on both mhsa q, v and ffn (qvmlp setting)
    """
    for name, layer in model.named_children():
        if 'attention' in name:
            layer.attention.neat_mode = neat_mode
            layer.attention.q_adapter = neat_multilayers(dim=mhsa_dim, depth=depth)
            layer.attention.v_adapter = neat_multilayers(dim=mhsa_dim, depth=depth)
            bound_method = forward_attn.__get__(layer.attention, layer.attention.__class__)
            setattr(layer.attention, 'forward', bound_method)
        elif 'dense' in name:
            if neat_mode == 1:
                continue
            layer.neat_mode = neat_mode
            layer.adapter = neat_multilayers(dim=ffn_dim, out_dim=layer.weight.shape[0])
            bound_method = forward_ffn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_neat_multilayers(layer, neat_mode, mhsa_dim, ffn_dim, depth=depth)

