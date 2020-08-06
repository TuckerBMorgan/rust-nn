
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
/*
pub struct Conv2dLayerConfig {
    pub stride: usize,
    pub kernal_size: (usize, usize),
    pub number_of_filters: usize,
    pub zero_padding: bool
}

impl Conv2dLayerConfig {
    pub fn new(stride: usize, kernal_size: (usize, usize), number_of_filters: usize, zero_padding: bool) -> Conv2dLayerConfig {
        Conv2dLayerConfig {
            stride,
            kernal_size,
            number_of_filters,
            zero_padding
        }
    }
}
*/
/*
    Why we should look towards Magic the Gathering for how to build AI
    I think some AI is really fucking boring
    AIs that do one, Detect faces, drive a car, "Narrow AIs" is the normal term
    These won't change the world at the end of the day. 
    Why?
    Because they fundementally are just more flexable software, given enough man hours
    and money you could write code could do what they do 
    I am much more curious about AGI, Artifical General Intelligence
    I think it is a question of when not if at this point.
    At that point we must ourselves the question, in what form do we want a AI to take?
    The goal of AGI as I see it, is to make an AI that is as Human as possible. 
    I guess at the end of the day you can think of it as a few steps
    we have Narrow AI, good at a single task, then you can tie a bunch of those
    together, and have what satisfiies DeepMinds Definition of Intelligence 
    "Intelligence measures an agent’s ability to achieve goals in a wide range of environments”"
    Now to be fair I don't think that this is a bad definition, it does a great job of defining 
    the complexity of intelligence, it isnt just being great at one thing, it is being great
    at many. But I think you can also cheat into something that looks like it, a Chinese Room 
    solution if you will. You can squint and say that it works, but no one would walk away feeling
    satisified
*/