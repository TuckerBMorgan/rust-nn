use ndarray::prelude::*;
use ndarray::Array;
use std::f32;
use rand::Rng;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::{
    DenseLayerConfig, InputLayerConfig,
    ActivationFunction,
    LINEAR_FUNCTION, RELU_FUNCTION, SIGMOID_FUNCTION, 
    ActivationPair
};

pub struct LayerLayout {
    weights: Array::<f32, Dim<[usize; 2]>>,
    activation_function: ActivationPair
}

impl LayerLayout {
    pub fn new(weights: Array::<f32, Dim<[usize; 2]>>, activation_function: ActivationPair) -> LayerLayout {
        LayerLayout {
            weights,
            activation_function
        }
    }
}

pub struct CompiledNetwork {
    layers_layout: Vec<LayerLayout>
}

impl CompiledNetwork {
    pub fn new (layers_layout: Vec<LayerLayout>) -> CompiledNetwork {
        CompiledNetwork {
            layers_layout
        }
    }

    pub fn forward_pass(&mut self, input_data: &Array::<f32, Dim<[usize; 2]>>) -> Vec<Array::<f32, Dim<[usize; 2]>>> {
        let mut layer_input = Array::<f32, _>::zeros((1, input_data.shape()[1]).f());
        layer_input.assign(&input_data);
        let mut outputs = vec![];
        
        //The first layer is an input layer, it is an artifact of the way I create the matrixs, need to think about how to remove it
        for layer in self.layers_layout.iter_mut() {
            let pre_output = layer_input.dot(&layer.weights);
            let final_ouput = pre_output.map(layer.activation_function.0);
            //TODO: This is really expensive, would prefer to find a way to not double the work here
            let mut array_copy = Array::<f32, _>::zeros((1, final_ouput.shape()[1]).f());
            array_copy.assign(&final_ouput);
            outputs.push(array_copy);
            let mut foo = Array::<f32, _>::zeros((1, final_ouput.shape()[1]).f());
            foo.assign(&final_ouput);
            layer_input = foo;
        }
    
        return outputs;
    }

    pub fn backprop(&mut self, expected_outputs : &Array::<f32, Dim<[usize; 2]>>, actual_outputs: &Vec<Array::<f32, Dim<[usize; 2]>>>) -> Vec<Array::<f32, Dim<[usize; 2]>>> {
        let index = &actual_outputs.len() - 1;
        let mut error = expected_outputs - &actual_outputs[index];
        let mut all_deltas = vec![];
    
        for (i, layer) in self.layers_layout.iter().rev().enumerate() {
            let output_derivative = actual_outputs[(actual_outputs.len() - 1) - i].map(layer.activation_function.1);

            let deltas = &error * &output_derivative;
            error = layer.weights.dot(&deltas.t()).reversed_axes();
            all_deltas.push(deltas);
        }
        all_deltas.reverse();
        return all_deltas;
    }

    pub fn update_weights(&mut self, deltas : Vec<Array::<f32, Dim<[usize; 2]>>>, inputs : &Array::<f32, Dim<[usize; 2]>>, learning_rate: f32, outputs: &Vec<Array::<f32, Dim<[usize; 2]>>>) {
        let mut use_input = inputs;
        for i in 0..self.layers_layout.len() {
            self.layers_layout[i].weights = &self.layers_layout[i].weights + &deltas[i].map(|x|x * learning_rate).reversed_axes().dot(use_input).reversed_axes();
            use_input = &outputs[i];
        }
    }
    
    pub fn train(&mut self, inputs: Vec<Array::<f32, Dim<[usize; 2]>>>, outputs: Vec<Array::<f32, Dim<[usize; 2]>>>, epochs: usize) {
        let mut rng = rand::thread_rng();
        for e in 0..epochs {
           // println!("Epoch {}", e);

            //Run a batch of training
            for i in 0..inputs.len() {                
                let index = rng.gen_range(0, inputs.len());
                let run_input = &inputs[index];
                let run_output = &outputs[index];
                let forward_output = self.forward_pass(run_input);
                let deltas = self.backprop(run_output, &forward_output);
                self.update_weights(deltas, run_input, 0.001f32, &forward_output);
            }

            //Calculate error
            if e % 100 == 0 {
                let mut error = 0.0f32;
                for n in 0..inputs.len() {
                    let test_output = self.forward_pass(&inputs[n]);
                    let final_ouput = &test_output[test_output.len() - 1];
                    error += (&outputs[n] - final_ouput).mean().unwrap();
                }
                println!("Error for Epoch {} is {}", e, error);
            }            
        }
    }
}

pub struct Network {
    layers_layout: Vec<LayerLayout>
}

impl Network {
    pub fn new() -> Network {
        Network {
            layers_layout: vec![],
        }
    }

    pub fn add_input_layer(mut self, input_layer_config: InputLayerConfig) -> Network {
        if self.layers_layout.len() != 0 {
            panic!("This network already has a input layer");
        }
        let input_array = Array::<f32, _>::from_elem((1, input_layer_config.number_of_inputs), 1.);
        
        let layer = LayerLayout::new(input_array, LINEAR_FUNCTION);
        self.layers_layout.push(layer);
        self
    }

    pub fn add_dense_layer(mut self, dense_layer_config: DenseLayerConfig) -> Network {
        
        //https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_uniform
        let le_cun_random_limit = (3.0f32 / dense_layer_config.number_of_nerons as f32).sqrt();
        
        let previous_array_ouput = self.layers_layout[self.layers_layout.len() - 1].weights.shape()[1];
        let next_array = Array::<f32, _>::random((previous_array_ouput, dense_layer_config.number_of_nerons), Uniform::new(-le_cun_random_limit, le_cun_random_limit));
        
        let activation_function;

        match dense_layer_config.activation_function {
            ActivationFunction::linear => {
                activation_function = LINEAR_FUNCTION;
            },
            ActivationFunction::sigmoid => {
                activation_function = SIGMOID_FUNCTION;
            },
            ActivationFunction::relu => {
                activation_function = RELU_FUNCTION;
            }
        }
        let layer = LayerLayout::new(next_array, activation_function);

        self.layers_layout.push(layer);
        self
    }

    pub fn compile(mut self) -> CompiledNetwork {
        let _ = self.layers_layout.remove(0);//Remove the dummy input layer
        let mut complied_network = CompiledNetwork::new(self.layers_layout);
        return complied_network;
    }    
}
