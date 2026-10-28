use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(hidden: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Attention {
            q: linear(hidden, hidden, vb.pp("q_proj"))?,
            k: linear(hidden, hidden, vb.pp("k_proj"))?,
            v: linear(hidden, hidden, vb.pp("v_proj"))?,
            out: linear(hidden, hidden, vb.pp("out_proj"))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q.forward(x)?)?;
        let k = split(self.k.forward(x)?)?;
        let v = split(self.v.forward(x)?)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, self.heads * self.head_dim))?;
        self.out.forward(&ctx)
    }
}

pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Mlp {
            fc1: linear(hidden, intermediate, vb.pp("fc1"))?,
            fc2: linear(intermediate, hidden, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

pub struct EncoderLayer {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        heads: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(EncoderLayer {
            ln1: layer_norm(hidden, eps, vb.pp("layer_norm1"))?,
            attn: Attention::new(hidden, heads, vb.pp("self_attn"))?,
            ln2: layer_norm(hidden, eps, vb.pp("layer_norm2"))?,
            mlp: Mlp::new(hidden, intermediate, vb.pp("mlp"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = (x + self.attn.forward(&self.ln1.forward(x)?, mask)?)?;
        &h + self.mlp.forward(&self.ln2.forward(&h)?)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn block_keeps_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let layer = EncoderLayer::new(16, 32, 4, 1e-6, vb).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 5, 16), &dev).unwrap();
        let y = layer.forward(&x, None).unwrap();
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn masked_tokens_get_no_attention() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let attn = Attention::new(8, 2, vb).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 3, 8), &dev).unwrap();
        let mask = Tensor::from_vec(vec![0f32, 0.0, -1e9], (1, 1, 1, 3), &dev).unwrap();
        let full = attn.forward(&x, Some(&mask)).unwrap();
        let mut changed = x.to_vec3::<f32>().unwrap();
        changed[0][2] = vec![9.0; 8];
        let x2 = Tensor::new(changed, &dev).unwrap();
        let full2 = attn.forward(&x2, Some(&mask)).unwrap();
        let a = full.narrow(1, 0, 2).unwrap().to_vec3::<f32>().unwrap();
        let b = full2.narrow(1, 0, 2).unwrap().to_vec3::<f32>().unwrap();
        for (ra, rb) in a[0].iter().zip(&b[0]) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-4);
            }
        }
    }
}
