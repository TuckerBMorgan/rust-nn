
struct Logit {
    weights: Vec<f32>
}

impl Logit {
    
    pub fn new(input_size: usize) -> Logit {
        Logit {
            weights: vec![]
        }
    }

    pub fn fire(&self, inputs: &Vec<f32>) -> f32 {
        let mut sum = 0.0;

        for (i, v) in inputs.iter().enumerate() {
            sum += v * self.weights[i];
        }

        let e_to_neg_x = (-sum).exp();
        return 1.0 / ( 1.0 + e_to_neg_x);
    }

}

struct LayerConfig {
    input_size: usize,
    number_of_logits: usize
}

struct Layer {
    logits: Vec<Logit>,
    input_size: usize
}

impl Layer {
    pub fn new(layer_config: LayerConfig) -> Layer {
        let mut logits = vec![];

        for i in 0..layer_config.number_of_logits {
            let l = Logit::new(layer_config.input_size);
            logits.push(l)
        }

        Layer {
            logits,
            input_size: layer_config.input_size
        }
    }

    pub fn fire(&self, input: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if input.len() != self.input_size {
            return Err("Size of data input does not match size of input for layer");
        }

        let mut outputs : Vec<f32> = vec![];
        for l in &self.logits {
            let t = l.fire(&input);
            outputs.push(t);
        }

        return Ok(outputs);
    }
}


struct Network {
    layers: Vec<Layer>,
    input_layer_size: usize
}

impl Network {
    pub fn new(layer_config: Vec<LayerConfig>) -> Network {
        let mut layers = vec![];
        let input_size = layer_config[0].input_size;
        for lc in layer_config {
            layers.push(Layer::new(lc));
        }

        Network {
            layers,
            input_layer_size: input_size
        }
    }

    pub fn fire(&self, input_data: Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if input_data.len() != self.input_layer_size {
            return Err("Differeance in input size to the network///TODO LET THE LAYER HANDLE THIS")
        }
        let pass_data = &input_data;
        for l in &self.layers {
            pass_data = l.fire(pass_data)
        }
        return Ok(pass_data.to_vec());
    }
}

fn main() {

    let mut lc = vec![];

    for i in 0..10 {
        lc.push(LayerConfig {
            input_size: 10,
            number_of_logits: 10
        });
    }

    let n  = Network::new(lc);
}
