extern crate rand;

use rand::Rng;

struct Logit<'a> {
    weights: Vec<f32>,
    last_output: f32,
    delta: f32,
    activation_function: &'a dyn Fn(f32) -> f32,
    inverse_activation_function: &'a dyn Fn(f32) -> f32
}

impl<'a> Logit<'a> {
    
    pub fn new(input_size: usize,
               activation_function: &'a dyn Fn(f32) -> f32,
               inverse_activation_function: &'a dyn Fn(f32) -> f32) 
                    -> Logit<'a> {

        let mut weights = vec![];
        let mut rng = rand::thread_rng();

        for _ in 0..input_size {
            weights.push(rng.gen::<f32>());
        }

        Logit {
            weights,
            last_output: 0.0,
            delta: 0.0,
            activation_function,
            inverse_activation_function
        }
    }

    pub fn fire(&mut self, inputs: &Vec<f32>) -> f32 {
        let mut sum = 0.0;

        for (i, v) in inputs.iter().enumerate() {
            //println!("weight {}", self.weights[i]);
            sum += v * self.weights[i];
        }
        self.last_output = (self.activation_function)(sum);
        return self.last_output;
    }
}

struct LayerConfig<'a> {
    input_size: usize,
    number_of_logits: usize,
    activation_function:&'a dyn Fn(f32) -> f32,
    inverse_activation_function: &'a dyn Fn(f32) -> f32
}

struct Layer<'a> {
    logits: Vec<Logit<'a>>,
    input_size: usize,
    activation_function: &'a dyn Fn(f32) -> f32,
    inverse_activation_function: &'a dyn Fn(f32) -> f32
}

impl<'a> Layer<'a> {
    pub fn new(layer_config: LayerConfig) -> Layer {
        let mut logits = vec![];

        for i in 0..layer_config.number_of_logits {
            let l = Logit::new(layer_config.input_size, layer_config.activation_function, layer_config.inverse_activation_function);
            logits.push(l)
        }

        Layer {
            logits,
            input_size: layer_config.input_size,
            activation_function: layer_config.activation_function,
            inverse_activation_function: layer_config.inverse_activation_function
        }
    }

    pub fn fire(&mut self, input: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if input.len() != self.input_size {
            return Err("Size of data input does not match size of input for layer");
        }

        let mut outputs : Vec<f32> = vec![];
        for l in self.logits.iter_mut() {
            let t = l.fire(&input);
            outputs.push(t);
        }

        return Ok(outputs);
    }

}


struct Network<'a> {
    layers: Vec<Layer<'a>>,
    input_layer_size: usize
}

impl<'a> Network<'a> {
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

    pub fn fire(&mut self, input_data: &Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if input_data.len() != self.input_layer_size {
            return Err("Differeance in input size to the network///TODO LET THE LAYER HANDLE THIS")
        }

        let mut pass_data = Ok(input_data.clone());
        for l in self.layers.iter_mut() {
            let result = l.fire(&pass_data.unwrap());
            match result {
                Err(e) => {
                    return Err(e);
                },
                Ok(_) => {
                    pass_data = result;
                }

            }
        }
        return pass_data;
    }

    //https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    pub fn backprob(&mut self, output: &Vec<f32>, expected: &Vec<f32>)  {

        let mut error = 0.0f32;
        let number_of_layers = self.layers.len();
        //scratch pad??????
        let mut errors = vec![];

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            if i == number_of_layers - 1 {
                for j in 0..layer.logits.len() {
                    errors.push(expected[j] - layer.logits[i].last_output);
                }
                let mut error_sum = 0.0f32;
                for e in &errors {
                    error_sum += e;
                }
               // println!("error {}", error_sum);
            }
            for (j, log) in layer.logits.iter_mut().enumerate() {
                log.delta = errors[j] * (log.inverse_activation_function)(log.last_output);
            }
            errors = vec![];
            for j in 0..layer.logits.len() {
                error = 0.0f32;
                for n in 0..layer.logits.len() { 
                    let logit = &layer.logits[n];
                    error += logit.weights[j] * logit.delta;
                }
                errors.push(error);
            }
        }
    }
    pub fn update_weights(&mut self, pass_inputs : &Vec<f32>) {
        let mut inputs = vec![];
        for i in 0..self.layers.len() {

            if i == 0 {
                inputs = pass_inputs.to_vec();
            }

            for logit in self.layers[i].logits.iter_mut() {
                for j in 0..logit.weights.len() {
                    logit.weights[j] += 0.00001 * logit.delta * inputs[j];
                }
            }

            inputs = vec![];
            for logit in self.layers[i].logits.iter_mut() {
                inputs.push(logit.last_output);
            }
        }
    }
}

fn sigmoid(v: f32) -> f32 {
    let e_to_neg_x = (-v).exp();
    return 1.0 / ( 1.0 + e_to_neg_x);
}

fn sigmoid_dx(v: f32) -> f32 {
    return v * (1.0 - v);
}

fn linear(v: f32) -> f32 {
    return v;
}

fn linear_dx(v: f32) -> f32 {
    return -v;
}

fn relu(v: f32) -> f32 {
    if v > 0.0f32 {
        return v;
    }
    return 0f32;
}

fn relu_dx(v: f32) -> f32 {
    if v > 0.0f32 {
        return 1.0f32;
    }
    else if v == 0.0f32 {
        return 0.5f32;
    }
    return 0.0f32;
}

fn main() {


    let mut lc = vec![];

    lc.push(LayerConfig {
        input_size: 1,
        number_of_logits: 1,
        activation_function: &relu,
        inverse_activation_function: &relu_dx
    });


    let mut n  = Network::new(lc);
    let data_length = 25;
    let mut fake_input = vec![];
    for i in 0..data_length  {
        fake_input.push(i as f32);
    }

    let mut foo = vec![];
    for i in 0..data_length {
        foo.push((i * 2)as f32 );
    }
    
    let mut rng = rand::thread_rng();

    let mut i = 0;
    loop {
        let mut index = rng.gen_range(0, data_length);
        let fake_fake_input = vec![fake_input[index]];
        let fake_fake_output = vec![foo[index]];
        let result = n.fire(&fake_fake_input).unwrap();
        let mut mean_sqaure_error = 0.0f32;
        if i % 100 == 0 {
            let mut mean_sqaure_error = 0.0f32;
            for i in 0..data_length {
                let result = n.fire(&vec![i as f32]).unwrap();
                mean_sqaure_error += ((i * 2) as f32 - result[0]).powf(2.0);
            }
            println!("mse {}", mean_sqaure_error / data_length as f32);
        }

        n.backprob(&result, &fake_fake_output);
        n.update_weights(&fake_fake_input);
        i += 1;
    }
}
