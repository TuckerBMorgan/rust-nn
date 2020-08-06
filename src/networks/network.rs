use ndarray::prelude::*;
use ndarray::{Array, IxDyn};
use std::f32;
use rand::Rng;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::*;


pub struct CompiledNetwork {
    layers_layout: Vec<Box<LayerV2>>
}

impl CompiledNetwork {
    pub fn new (layers_layout: Vec<Box<LayerV2>>) -> CompiledNetwork {
        CompiledNetwork {
            layers_layout
        }
    }

    pub fn forward_pass(&mut self, input_data: &Array::<f32, IxDyn>) -> Vec<Array::<f32, IxDyn>> {
        let mut layer_input = Array::<f32, _>::zeros((1, input_data.shape()[1]).f());
        layer_input.assign(&input_data);
        let mut outputs = vec![];
        
        //The first layer is an input layer, it is an artifact of the way I create the matrixs, need to think about how to remove it
        for layer in self.layers_layout.iter_mut() {
            let final_ouput = layer.forward(layer_input.into_dyn());
            let mut array_copy = Array::<f32, _>::zeros((1, final_ouput.shape()[1]).f());
            array_copy.assign(&final_ouput);
            outputs.push(array_copy);
            let mut foo = Array::<f32, _>::zeros((1, final_ouput.shape()[1]).f());
            foo.assign(&final_ouput);
            layer_input = foo;
        }
    
        return outputs.iter_mut().map(|x| x.to_owned().into_dyn()).collect();
    }
    
    pub fn backprop(&mut self, expected_outputs : &Array::<f32, IxDyn>, actual_outputs: &Vec<Array::<f32, IxDyn>>) -> Vec<Array::<f32, IxDyn>> {
        let index = &actual_outputs.len() - 1;
        let mut error = expected_outputs - &actual_outputs[index];
        let mut all_deltas = vec![];
    
        for (i, layer) in self.layers_layout.iter_mut().rev().enumerate() {
            let (a, b) = layer.backprop(&error, &actual_outputs[(actual_outputs.len() - 1) - i]);
            error = b;
            all_deltas.push(a);
        }
        all_deltas.reverse();
        return all_deltas;
    }
    
    pub fn update_weights(&mut self, deltas : Vec<Array::<f32, IxDyn>>, inputs : &Array::<f32, IxDyn>, learning_rate: f32, outputs: &Vec<Array::<f32, IxDyn>>) {
        let mut use_input = inputs;
        for (i, layer) in self.layers_layout.iter_mut().enumerate() {
            layer.update_weights(use_input, &deltas[i], 0.001f32);
            use_input = &outputs[i];
        }
    }
    
    pub fn train(&mut self, inputs: Vec<Array::<f32, IxDyn>>, outputs: Vec<Array::<f32, IxDyn>>, epochs: usize) {
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

pub struct NetworkConfig {
    layers: Vec<Box<LayerV2>>
}

impl NetworkConfig {
    pub fn new() -> NetworkConfig {
        NetworkConfig {
            layers: vec![]
        }
    }

    pub fn add_input_layer(mut self, input_layer_config: InputLayerConfig) -> NetworkConfig {
        if self.layers.len() != 0 {
            panic!("This network already has a input layer");
        }
        let input_array = Array::<f32, _>::from_elem((1, input_layer_config.number_of_inputs), 1.);
        
        let layer = InputLayer::new(input_array.into_dyn());
        self.layers.push(Box::new(layer));
        self
    }

    pub fn add_dense_layer(mut self, dense_layer_config: DenseLayerConfig) -> NetworkConfig {
        
        //https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_uniform
        let le_cun_random_limit = (3.0f32 / dense_layer_config.number_of_nerons as f32).sqrt();
        
        let previous_array_ouput = self.layers[self.layers.len() - 1].shape()[1];
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
        let layer = DenseLayer::new(next_array.into_dyn(), activation_function);

        self.layers.push(Box::new(layer));
        self
    }
    /*
    pub fn add_conv2d_layer(mut self, conv2d_layer: Conv2dLayerConfig) -> NetworkConfig {
        
    }
*/
    pub fn compile(mut self) -> CompiledNetwork {
        //removing the input layer which is just there to give a shape for the first layer
        self.layers.remove(0);
        let mut compiled_network = CompiledNetwork::new(self.layers);
        return compiled_network;
    }
}


/*
Complexity is for the lazy
A universial truth I have found is that we arrive only at complexity because we are too lazy to find better simple solution
In all the code bases I have worked in the worst parts where always the ones in which someone, or someones had tried to 
flex their "genius", or found some slightly more manual solution to a problem instead of taking the time to find a simple soultion
we should build with Lego not with 

*/