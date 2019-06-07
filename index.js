const brain = require('brain.js');
const brain_bow = require('brain-bow');

const State = {
    UNTRAINED: "UNTRAINED",
    TRAINED: "TRAINED",
    OUTDATED: "OUTDATED",
    TRAINING: "TRAINING"
}

// _net is the Artificial Neural Network
const _net = new brain.NeuralNetwork();
// _bow is a library intended to make bag of words processing,
// for example vectorizing strings
const _bow = new brain_bow.BagOfWords();
// _dict is a dictionary build from all the sentences of input data texts
let _dict = {};
// _classes are all the classes from input data text formatted to feed the ANN
/* _classes is like this:

    { apagar_lampara: 0, encender_lampara: 1 }
*/
let _classes = {};

let _status = State.UNTRAINED;

let _configuration = {
    iterations: 3000, // the maximum times to iterate the training data
    errorThresh: 0.006, // the acceptable error percentage from training data
    log: true, // true to use console.log, when a function is supplied it is used
    logPeriod: 10, // iterations between logging out
    learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
    momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
};

/**
 * Shuffle an array of objects
 * @param {*} a is an array of objects:
    // [
    //    {label: "encender_lampara", text: "enciende la luz"},
    //    ...
    //    {label: "apagar_lampara", text: "apaga la lámpara"} 
    // ]
 * 
 */
const shuffle = function (a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

/**
 * Build an array of objects from the input data string
 * each object is like this:
 * {label: "encender_lampara", text:  "enciende la luz"}
 * 
 * @param {*} inputDataString is an JSON string like this
 * {
        "encender_lampara": [
            "enciende la luz",
            "esto está muy oscuro"
        ],
        "apagar_lampara": [
            "apaga la luz",
            "apaga la lámpara"
        ]
   }
 */
const buildTrainDataFromInputDataString = function (inputDataString) {
    let inputDataObj = JSON.parse(inputDataString);
    let traindata = [];
    for (const key in inputDataObj) {
        for (const text of inputDataObj[key]) {
            traindata.push({ label: key, text: text })
        }
    }
    // now we shuffle traindata
    shuffle(traindata);

    return traindata;
}

/**
 * Prepare train data to be feed in brain.js Artificial Neural Network
 * 
 * @param {*} traindata  is an array of objects like this
 *  [
 *     {label: "encender_lampara", text: "enciende la luz"},
 *     ...
 *     {label: "apagar_lampara", text: "apaga la lámpara"} 
 *   ]
 */
const prepareTrainData = function (traindata) {
    let texts = [];
    let traindata_for_ann;

    // extract all the classe (which are the labels) without
    // repetition from traindata and map to number in order to
    // feed the ANN
    let i = 0;
    for (let data of traindata) {
        texts.push(data.text);
        if (_classes[data.label] == undefined) {
            _classes[data.label] = i;
            i++;
        }
    }

    // build dictionary
    _dict = _bow.extractDictionary(texts);

    // build training data to feed ANN
    /* 
    traindata_for_ann is like:
    [ { input: [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 ], output: [ 1, 0 ] },
      { input: [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 0 ], output: [ 1, 0 ] },
      { input: [ 0, 1, 0, 0, 0, 0, 0, 1, 1, 1 ], output: [ 0, 1 ] },
      { input: [ 0, 1, 1, 0, 0, 0, 0, 1, 0, 0 ], output: [ 0, 1 ] } ]
    */
    traindata_for_ann = [];
    for (let data of traindata) {
        let item = {
            input: _bow.bow(data.text, _dict),
            output: _bow.vec_result(_classes[data.label], Object.keys(_classes).length)
        }
        traindata_for_ann.push(item);
    }

    return traindata_for_ann;
}

//~~~~~~~~~~~ EXPORTED OBJECTS ~~~~~~~~~~~//

/**
 * Set a new configuration to training net
 * @param {*} config is like this:
 * {
 *  iterations: 3000, // the maximum times to iterate the training data
 *  errorThresh: 0.006, // the acceptable error percentage from training data
 *  log: true, // true to use console.log, when a function is supplied it is used
 *  logPeriod: 10, // iterations between logging out
 *  learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
 *   momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
 * };
 */
exports.setConfiguration = function(config){
    _configuration = config;
}

/**
 * Train the ANN with input data  JSON string.
 * 
 * Please pay attention!, this function returns a promise.
 * 
 * @param {*} modelJSON is a json string with the labels and texts
 * example of modelJSON:
 * {
 *     "encender_lampara": [
 *          "enciende la luz",
 *          "esto está muy oscuro"
 *      ],
 *      "apagar_lampara": [
 *          "apaga la luz",
 *          "apaga la lámpara"
 *      ]
 * }
 */
exports.train = function (modelJSON) {

    /* traindata is an array like this:
     [
        {label: "encender_lampara", text: "enciende la luz"},
        ...
        {label: "apagar_lampara", text: "apaga la lámpara"} 
     ]
     */

    const traindata = buildTrainDataFromInputDataString(modelJSON);

    const traindata_for_ann = prepareTrainData(traindata);

    console.log(traindata);
    console.log('############');
    console.log(traindata_for_ann);

    _status = State.TRAINING;
    let promise = _net.trainAsync(traindata_for_ann, _configuration).then(
        (result) => {
            _status = State.TRAINED;
            return result;
        }
    )

    return promise;
}

/**
 * Run the model to classify a text
 * 
 * @param {*} entry is a string which we want to classify 
 */
exports.run = function (entry) {
    if(_status == State.UNTRAINED){
        throw "Network UNTRAINED, can't make any prediction!"
    }
    // vectorize as a Bag Of Word
    let term = _bow.bow(entry, _dict);
    let predict = _net.run(term);
    let i = _bow.maxarg(predict);
    let flippedClasses = {};
    for (let key in _classes) {
        flippedClasses[_classes[key]] = key
    }

    let prediction = {};
    for (let i = 0; i < predict.length; i++) {
        prediction[flippedClasses[i]] = predict[i];
    }
    
    let result = {
        text: entry,
        label: flippedClasses[i],
        confidence: predict[i],
        prediction: prediction,
        status: _status
    }

    return result;
}