
pub enum ActivationFunction {
    linear,
    sigmoid,
    relu
}

pub struct InputLayerConfig {
    pub number_of_inputs: usize
}

impl InputLayerConfig {
    pub fn new(number_of_inputs: usize) -> InputLayerConfig {
        InputLayerConfig {
            number_of_inputs
        }
    }
}

pub struct DenseLayerConfig {
    pub number_of_nerons: usize,
    pub activation_function: ActivationFunction
}

impl DenseLayerConfig {
    pub fn new(number_of_nerons: usize, activation_function: ActivationFunction) -> DenseLayerConfig {
        DenseLayerConfig {
            number_of_nerons,
            activation_function
        }
    }
}