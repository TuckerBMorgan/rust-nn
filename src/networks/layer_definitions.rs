use ndarray::prelude::*;
use ndarray::{Array, IxDyn, Ix2};
use std::f32;
use rand::Rng;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::*;

pub trait LayerV2 {
    fn forward(&mut self, input: Array::<f32, IxDyn>) -> Array::<f32, IxDyn>;
    fn backprop(&mut self, error: &Array::<f32, IxDyn>, output: &Array::<f32, IxDyn>) -> (Array::<f32, IxDyn>, Array::<f32, IxDyn>);
    fn update_weights(&mut self, input: &Array::<f32, IxDyn>, deltas: &Array::<f32, IxDyn>, learning_rate: f32);
    fn shape(&self) -> &[usize];
}

pub struct InputLayer {
    weights: Array::<f32, Ix2>
}

impl InputLayer {
    pub fn new(weights: Array::<f32, IxDyn>) -> InputLayer {
        InputLayer {
            weights: weights.into_dimensionality::<Ix2>().unwrap()
        }
    }
}

impl LayerV2 for InputLayer {
    fn forward(&mut self, input: Array::<f32, IxDyn>) -> Array::<f32, IxDyn> {
        return input;
    }
    
    fn shape(&self) -> &[usize] {
        return self.weights.shape();
    }
    fn backprop(&mut self, error: &Array::<f32, IxDyn>, output: &Array::<f32, IxDyn>) -> (Array::<f32, IxDyn>, Array::<f32, IxDyn>) {
        //It is ok that this makes no sense, we remove the input layer before we compile the network
        //So we can just have it do dummy shit
        let mut foo = Array::<f32, _>::zeros((1, 1).f()).into_dyn();
        let mut bar = Array::<f32, _>::zeros((1, 1).f()).into_dyn();
        return (foo, bar);
    }

    fn update_weights(&mut self, input: &Array::<f32, IxDyn>, deltas: &Array::<f32, IxDyn>, learning_rate: f32) {

    }
}

pub struct DenseLayer {
    weights: Array::<f32, Ix2>,
    activation_function: ActivationPair    
}

impl DenseLayer {
    pub fn new(weights: Array::<f32, IxDyn>, activation_function: ActivationPair) -> DenseLayer {
        DenseLayer {
            weights: weights.into_dimensionality::<Ix2>().unwrap(),
            activation_function
        }
    }
}

impl LayerV2 for DenseLayer {
    fn forward(&mut self, input: Array::<f32, IxDyn>) -> Array::<f32, IxDyn> {
        let use_input = input.to_owned().into_dimensionality::<Ix2>().unwrap();
        let pre_output = use_input.dot(&self.weights);
        let final_ouput = pre_output.map(self.activation_function.0);
        return final_ouput.into_dyn();
    }

    fn shape(&self) -> &[usize] {
        return self.weights.shape();        
    }
    
    fn backprop(&mut self, error: &Array::<f32, IxDyn>, output: &Array::<f32, IxDyn>) -> (Array::<f32, IxDyn>, Array::<f32, IxDyn>) {
        let use_output = output.to_owned().into_dimensionality::<Ix2>().unwrap();
        let output_derivative = use_output.map(self.activation_function.1);
        let deltas = (error * &output_derivative).into_dimensionality::<Ix2>().unwrap();
        let new_error = &self.weights.dot(&deltas.t()).reversed_axes();
        return (deltas.into_dyn(), new_error.to_owned().into_dyn());
    }

    fn update_weights(&mut self, input: &Array::<f32, IxDyn>, deltas: &Array::<f32, IxDyn>, learning_rate: f32) {
        let use_input = input.to_owned().into_dimensionality::<Ix2>().unwrap();
        let use_deltas = input.to_owned().into_dimensionality::<Ix2>().unwrap();
        self.weights = &self.weights + &use_deltas.map(|x|x * learning_rate).reversed_axes().dot(&use_input).reversed_axes();
    }
}
/*
pub struct Conv2dLayer {
    stride: usize,
    filter_size: (usize, usize),
    number_of_filters: usize,
    filters: Vec<Array::<f32, Dim<[usize; 4]>>>,
    activation_function: ActivationPair
}

impl Conv2dLayer {
    pub fn new(stride: usize, filter_size: (usize, usize), number_of_filters: usize, 
               filters: Vec<Array::<f32, IxDyn>>,, activation_function: ActivationPair) -> Conv2dLayer {
            Conv2dLayer {
                stride,
                filter_size,
                number_of_filters,
                filters,
                activation_function
            }
    }
}

impl LayerV2 for Conv2dLayer {
    fn forward(&mut self, input: Array::<f32, IxDyn>) -> Array::<f32, Dim<[usize; 4]>> {
        //Zero pad the input array
        //Init the outuput array
        //Apply the filter which is a dot product
        //return the array

        //treat each filter as it an entry into this array
        let mut final_outputs = vec![];

        //For each filter
        for f in self.filters { 

            let mut i = 0;
            let mut unshapped_output = vec![0; input.shape()[0] * input.shape()[1]];
            let windows = input.windows((self.filter_size.0, self.filter_size.1));
            //Preform the convolutions
            for w in windows {
                let output = f.dot(w);
                unshapped_output[i] = output;
                i += 1;
            }
            let activated_unshapped_output = unshapped_output.map(self.activation_function.0);
            let reshaped_output = Array::from_shape_vec(input.shape(), activated_unshapped_output);
            final_outputs.push(reshaped_output);
        }

        return final_outputs;
    }

    fn shape(&self) -> &[usize] {
        return self.weights.shape();
    }
    
    fn backprop(&mut self, error: &Array::<f32, IxDyn>, output: &Array::<f32, IxDyn>) -> (Array::<f32, IxDyn>, Array::<f32, IxDyn>) {
        let output_derivative = output.map(self.activation_function.1);
        let deltas = error * &output_derivative;
        let new_error = self.weights.dot(&deltas.t()).reversed_axes();
        return (deltas, new_error);
    }

    fn update_weights(&mut self, input: &Array::<f32, IxDyn>, deltas: &Array::<f32, IxDyn>, learning_rate: f32) {
        self.weights = &self.weights + &deltas.map(|x|x * learning_rate).reversed_axes().dot(input).reversed_axes();
    }
}
*/