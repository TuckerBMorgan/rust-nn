pub type ActivationPair = (fn(&f32) -> f32, fn(&f32)->f32);
pub const E: f32 = 2.71828182845904523536028747135266250f32;
use std::cmp;

//Linear, for the plebs
pub fn linear(x: &f32) -> f32 {
    return *x;
}

pub fn linear_dx(x: &f32) -> f32 {
    return -(*x);
}


//sigmoid, for the boring
pub fn sigmoid(x: &f32) -> f32 {
    return 1.0f32 / (1.0f32 + E.powf(*x));
}

pub fn sigmoid_dx(x: &f32) -> f32 {
    return sigmoid(x) * (1.0f32 - sigmoid(x));
}


//relu for the spicy
pub fn relu(x: &f32) -> f32 {
    return (*x).max(0.0f32);
}

fn relu_dx(x: &f32) -> f32 {
    if *x > 0.0f32 {
        return 1.0f32;
    }
    else if *x == 0.0f32 {
        return 0.5f32;
    }
    return 0.0f32;
}

pub const SIGMOID_FUNCTION : ActivationPair = (sigmoid as fn(&f32) -> f32, sigmoid_dx as fn(&f32) -> f32);
pub const RELU_FUNCTION : ActivationPair = (relu as fn(&f32) -> f32, relu_dx as fn(&f32) -> f32);
pub const LINEAR_FUNCTION : ActivationPair = (linear as fn(&f32) -> f32, linear_dx as fn(&f32) -> f32);