extern crate rand;
pub mod networks;
use networks::*;

use ndarray::prelude::*;
use ndarray::{Array, IxDyn, Ix2};
//Denkata 
//電硬
//電脳硬化症
//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

pub struct Test {
    test: Vec<Array::<f32, IxDyn>>
}

impl Test {
    pub fn new() -> Test {
        Test {
            test: vec![]
        }
    }
}

fn main() {

    let mut k = Array::zeros((2, 3));
    k[[0, 0]] = 1.;
    let mut j = Array::zeros((2, 3, 4));
    j[[0, 0, 0]] = 1.;
    let mut test = Test::new();
    test.test.push(k.into_dyn());

    let network = NetworkConfig::new();

    let input_layer_condig = InputLayerConfig::new(2);
    let dense_layer_1 = DenseLayerConfig::new(2,  ActivationFunction::sigmoid);
    let dense_layer_2 = DenseLayerConfig::new(1,  ActivationFunction::sigmoid);

    let mut compiled_network = network
                               .add_input_layer(input_layer_condig)
                               .add_dense_layer(dense_layer_1)
                               .add_dense_layer(dense_layer_2)
                               .compile();
    
    
    let pre_input = vec![[2.7810836f32,2.550537003],
                            [1.465489372,2.362125076],
                            [3.396561688,4.400293529],
                            [1.38807019,1.850220317],
                            [3.06407232,3.005305973],
                            [7.627531214,2.759262235],
                            [5.332441248,2.088626775],
                            [6.922596716,1.77106367],
                            [8.675418651,-0.242068655],
                            [7.673756466,3.508563011]
                           ];
    let inputs : Vec<Array::<f32, IxDyn>> = pre_input.iter().map(|x| return Array::<f32, Dim<[usize; 2]>>::from_shape_vec((1, 2), x.to_vec()).unwrap().into_dyn()).collect();

    let pre_output = vec![[0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0]
                            ];
    let outputs : Vec<Array::<f32, IxDyn>> = pre_output.iter().map(|x| return Array::<f32, Dim<[usize; 2]>>::from_shape_vec((1, 1), x.to_vec()).unwrap().into_dyn()).collect();
    compiled_network.train(inputs, outputs, 20000);
}