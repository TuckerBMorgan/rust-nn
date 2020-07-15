use ndarray::prelude::*;
use ndarray::Array;

//電脳硬化症
//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

enum ActivationFunction {
    relu,
    linear,
    sigmoid
}

struct DenseLayer {
    pub weights: Array1::<f32>
}

impl DenseLayer {
    pub fn new(number_of_nerons: usize) -> DenseLayer {
        DenseLayer {
            weights: Array::<f32, _>::zeros((number_of_nerons).f())
        }
    }
}

struct InputLayer {
    number_of_inputs: usize
}

struct DenseConfig {
    number_of_nerons: usize
}

struct Layer {
    weights: Array::<f32, Dim<[usize; 2]>>,
    activation_function: ActivationFunction
}

impl Layer {
    pub fn new(weights: Array::<f32, Dim<[usize; 2]>>, activation_function: ActivationFunction) -> Layer {
        Layer {
            weights,
            activation_function
        }
    }
}


fn compile_network(input_layer: InputLayer, layers_config: Vec<DenseConfig>) -> Vec<Layer> {

    let input_array = Array::<f32, _>::zeros((1, input_layer.number_of_inputs).f());
    let mut previous_array_ouput = input_array.shape()[1];
    println!("{:?}", previous_array_ouput);
    let input_layer = Layer::new(input_array, ActivationFunction::linear);

    let mut layers = vec![];

    for layer_config in layers_config {
        let next_array = Array::<f32, _>::zeros((previous_array_ouput, layer_config.number_of_nerons).f());
        previous_array_ouput = next_array.shape()[1];
        let layer = Layer::new(next_array, ActivationFunction::relu);
        layers.push(layer);
    }

    return layers;
}

fn forward_pass(input_data: Array::<f32, Dim<[usize; 2]>>, network: &mut Vec<Layer>) -> Vec<Array::<f32, Dim<[usize; 2]>>> {
    let mut layer_input = Array::<f32, _>::zeros((1, input_data.shape()[1]).f());
    layer_input.assign(&input_data);
    let mut outputs = vec![];

    for layer in network.iter_mut() {
        let pre_output = layer_input.dot(&layer.weights);
        let final_ouput = pre_output.map(|x| {if *x < 0.0f32 {return 0.0f32;} return *x});
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

fn backprop(expected_outputs:Array::<f32, Dim<[usize; 2]>>, outputs: Vec<Array::<f32, Dim<[usize; 2]>>>, network: &mut Vec<Layer>) -> Vec<Array::<f32, Dim<[usize; 2]>>> {
    let index = &outputs.len() - 1;
    let mut error = &expected_outputs - &outputs[index];
    let mut all_deltas = vec![];

    for (i, layer) in network.iter().skip(1).rev().enumerate() {
        let deltas = &error * &outputs[i].map(|x|{
                //derivative of the relu function
                if *x > 0.0f32 {
                    return 1.0f32;
                } 
                else if  *x == 0.0f32 { 
                    return 0.5f32;
                } 
                return 0.0f32;
            }
        );
        error = deltas.dot(&layer.weights);
        all_deltas.push(deltas);
    }
    return all_deltas;
}

fn main() {
    let input_layer = InputLayer {
        number_of_inputs: 3
    };

    let dense_config = DenseConfig {
        number_of_nerons: 2
    };

    let dense_config1 = DenseConfig {
        number_of_nerons: 2
    };

    let dense_config2 = DenseConfig {
        number_of_nerons: 2
    };

    let number_of_inputs = input_layer.number_of_inputs;
    let mut network = compile_network(input_layer, vec![dense_config, dense_config1, dense_config2]);
    let fake_input = Array::<f32, _>::zeros((1, number_of_inputs).f());
    let outputs = forward_pass(fake_input, &mut network);
    let mut fake_output = Array::<f32, _>::zeros((1, 2).f());
    let deltas = backprop(fake_output, outputs, &mut network);
}