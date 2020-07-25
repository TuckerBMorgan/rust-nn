mod layer_configs;
pub use layer_configs::{
    InputLayerConfig,
    DenseLayerConfig,
    ActivationFunction
};

mod network;
pub use network::Network;

mod activation;
pub use activation::*;

/*
    The importance of boredom
    I think some personality types need to get bored
    Why?
    Because it is the fear of boredom that will drive them to to do great things
    Boredom is akin to depression, a listless wasteland of emotion that sits
    in your memory.
    Unlike depression boredom is less sadness and more, nothing. No emotion,
    no motivation, just a voice in the back your head screaming at you DO
    WORK, CREATE, BUILD and then another voice saying Where do I start
    The only thing I have found that works is to just start, write a line of
    code, read a sentence in a book, anything, anything is better then boredom
*/