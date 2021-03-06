const BrainText = require('../index');

const modelJSON = '{"encender_lampara": ["enciende la luz","esto está muy oscuro"],"apagar_lampara": ["apaga la luz","apaga la lámpara"]}';

const config = {
    iterations: 3000, // the maximum times to iterate the training data
    errorThresh: 0.006, // the acceptable error percentage from training data
    log: true, // true to use console.log, when a function is supplied it is used
    logPeriod: 10, // iterations between logging out
    learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
    momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
};

let brainText = new BrainText();

// This line throw an error since network is untrainned yet
//let r = brain_text.run("encender luz");

// Change default configuration
console.log("######## Change configuration ########");
brainText.setConfiguration(config);

// Load data from JSON string
brainText.loadTrainDataFromInputDataString(modelJSON);

// Train the network and then run model to classify a text
brainText.train().then(() => {
    console.log("######## Train model and then run model ########");
    let r = brainText.run("encender luz");
    console.log(r);
});

// Run model to classify a text while net is being trained. Just for illustration
// purposes, this should'n be done.
setTimeout(() => {
    console.log("######## Run model while is being trained ########");
    let r = brainText.run("encender luz");
    console.log(r);
}, 200);

setTimeout(() => {
    brainText.addData([{label: 'encender_lampara', text: 'dale a la lamparita'}]);

    brainText.train().then(() => {
        let r = brainText.run('dale a la lamparita');
        console.log(r);
        r = brainText.run('encender luz');
        console.log(r);
    });
}, 2000)

console.log(brainText.toJSON());