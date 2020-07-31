use ndarray::prelude::*;
use ndarray::Array;
use std::f32;
use rand::Rng;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::*;

pub trait LayerV2 {
    fn forward(&mut self, input: Array::<f32, Dim<[usize; 2]>>) -> Array::<f32, Dim<[usize; 2]>>;
    fn backprop(&mut self, error: &Array::<f32, Dim<[usize; 2]>>, output: &Array::<f32, Dim<[usize; 2]>>) -> (Array::<f32, Dim<[usize; 2]>>, Array::<f32, Dim<[usize; 2]>>);
    fn update_weights(&mut self, input: &Array::<f32, Dim<[usize; 2]>>, deltas: &Array::<f32, Dim<[usize; 2]>>, learning_rate: f32);
    fn shape(&self) -> &[usize];
}

pub struct InputLayer {
    weights: Array::<f32, Dim<[usize; 2]>>
}

impl InputLayer {
    pub fn new(weights: Array::<f32, Dim<[usize; 2]>>) -> InputLayer {
        InputLayer {
            weights
        }
    }
}

impl LayerV2 for InputLayer {
    fn forward(&mut self, input: Array::<f32, Dim<[usize; 2]>>) -> Array::<f32, Dim<[usize; 2]>> {
        return input;
    }
    
    fn shape(&self) -> &[usize] {
        return self.weights.shape();
    }
    fn backprop(&mut self, error: &Array::<f32, Dim<[usize; 2]>>, output: &Array::<f32, Dim<[usize; 2]>>) -> (Array::<f32, Dim<[usize; 2]>>, Array::<f32, Dim<[usize; 2]>>) {
        //It is ok that this makes no sense, we remove the input layer before we compile the network
        //So we can just have it do dummy shit
        let mut foo = Array::<f32, _>::zeros((1, 1).f());
        let mut bar = Array::<f32, _>::zeros((1, 1).f());
        return (foo, bar);
    }

    fn update_weights(&mut self, input: &Array::<f32, Dim<[usize; 2]>>, deltas: &Array::<f32, Dim<[usize; 2]>>, learning_rate: f32) {

    }
}

pub struct DenseLayer {
    weights: Array::<f32, Dim<[usize; 2]>>,
    activation_function: ActivationPair    
}

impl DenseLayer {
    pub fn new(weights: Array::<f32, Dim<[usize; 2]>>, activation_function: ActivationPair) -> DenseLayer {
        DenseLayer {
            weights,
            activation_function
        }
    }
}

impl LayerV2 for DenseLayer {
    fn forward(&mut self, input: Array::<f32, Dim<[usize; 2]>>) -> Array::<f32, Dim<[usize; 2]>> {
        let pre_output = input.dot(&self.weights);
        let final_ouput = pre_output.map(self.activation_function.0);
        return final_ouput;
    }

    fn shape(&self) -> &[usize] {
        return self.weights.shape();        
    }
    
    fn backprop(&mut self, error: &Array::<f32, Dim<[usize; 2]>>, output: &Array::<f32, Dim<[usize; 2]>>) -> (Array::<f32, Dim<[usize; 2]>>, Array::<f32, Dim<[usize; 2]>>) {
        let output_derivative = output.map(self.activation_function.1);
        let deltas = error * &output_derivative;
        let new_error = self.weights.dot(&deltas.t()).reversed_axes();
        return (deltas, new_error);
    }

    fn update_weights(&mut self, input: &Array::<f32, Dim<[usize; 2]>>, deltas: &Array::<f32, Dim<[usize; 2]>>, learning_rate: f32) {
        self.weights = &self.weights + &deltas.map(|x|x * learning_rate).reversed_axes().dot(input).reversed_axes();
    }
}

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
