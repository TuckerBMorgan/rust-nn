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

    let mut layers = vec![];

    for layer_config in layers_config {
        let next_array = Array::<f32, _>::zeros((previous_array_ouput, layer_config.number_of_nerons).f());
        previous_array_ouput = next_array.shape()[1];
        let layer = Layer::new(next_array, ActivationFunction::relu);
        layers.push(layer);
    }

    return layers;
}

fn forward_pass(input_data: &Array::<f32, Dim<[usize; 2]>>, network: &mut Vec<Layer>) -> Vec<Array::<f32, Dim<[usize; 2]>>> {
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

fn backprop(expected_outputs:Array::<f32, Dim<[usize; 2]>>, outputs: &Vec<Array::<f32, Dim<[usize; 2]>>>, network: &mut Vec<Layer>) -> Vec<Array::<f32, Dim<[usize; 2]>>> {
    let index = &outputs.len() - 1;
    let mut error = &expected_outputs - &outputs[index];
    let mut all_deltas = vec![];

    for (i, layer) in network.iter().rev().enumerate() {
        let output_derivative = outputs[(outputs.len() - 1) - i].map(|x|{
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


        let deltas = &error * &output_derivative;

        error = layer.weights.dot(&deltas.t()).reversed_axes();
        all_deltas.push(deltas);
    }
    all_deltas.reverse();
    return all_deltas;
}


fn update_weights(network: &mut Vec<Layer>, deltas : Vec<Array::<f32, Dim<[usize; 2]>>>, inputs : &Array::<f32, Dim<[usize; 2]>>, learning_rate: f32, outputs: &Vec<Array::<f32, Dim<[usize; 2]>>>) {
    let mut use_input = inputs;
    for i in 0..network.len() {
        println!("{:?}", use_input.shape());
        println!("{:?}", deltas[i].shape());
        println!("{:?}", network[i].weights.shape());
        network[i].weights = &network[i].weights + &deltas[i].map(|x|x * learning_rate).reversed_axes().dot(use_input).reversed_axes();
        use_input = &outputs[i];
    }
}

fn main() {
    let input_layer = InputLayer {
        number_of_inputs: 3
    };

    let dense_config = DenseConfig {
        number_of_nerons: 2
    };

    let dense_config1 = DenseConfig {
        number_of_nerons: 4
    };

    let dense_config2 = DenseConfig {
        number_of_nerons: 2
    };

    let dense_config3 = DenseConfig {
        number_of_nerons: 5
    };


    let number_of_inputs = input_layer.number_of_inputs;
    let mut network = compile_network(input_layer, vec![dense_config, dense_config1, dense_config2, dense_config3]);
    let fake_input = Array::<f32, _>::zeros((1, number_of_inputs).f());
    let outputs = forward_pass(&fake_input, &mut network);

    let mut fake_output = Array::<f32, _>::zeros((1, 5).f());
    let deltas = backprop(fake_output, &outputs, &mut network);
   // for d in deltas {
        //println!("{:?}", d.shape());
  //  }
    //return;
    update_weights(&mut network, deltas, &fake_input, 0.001f32, &outputs);
}