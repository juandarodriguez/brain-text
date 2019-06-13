const brain = require('brain.js');
const brain_bow = require('brain-bow');

const State = {
    UNTRAINED: "UNTRAINED",
    TRAINED: "TRAINED",
    OUTDATED: "OUTDATED",
    TRAINING: "TRAINING"
}

/**
 * Shuffle an array of objects
 * @param {*} a is an array of objects:
 *  [
 *      {label: "encender_lampara", text: "enciende la luz"},
 *      ...
 *      {label: "apagar_lampara", text: "apaga la lámpara"} 
 *  ]
 * 
 */
const shuffle = function (a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

const buildClassesAndDict = function (traindata, bow) {

    let texts = [];
    let classes = {};
    // extract all the classe (which are the labels) without
    // repetition from traindata and map to number in order to
    // feed the ANN
    let i = 0;
    for (let data of traindata) {
        texts.push(data.text);
        if (classes[data.label] == undefined) {
            classes[data.label] = i;
            i++;
        }
    }

    let dict = bow.extractDictionary(texts);

    return { classes: classes, texts: texts, dict: dict };
}

/////// EXPORTED CONSTRUCTOR FUNCTION /////////

function BrainText() {

    /*
     * An array of object like this:
     * [
     *     {label: "encender_lampara", text: "enciende la luz"},
     *     ...
     *     {label: "apagar_lampara", text: "apaga la lámpara"} 
     *   ]
     */
    this._traindata = [];
    // _net is the Artificial Neural Network
    this._net = new brain.NeuralNetwork();
    // _bow is a library intended to make bag of words processing,
    // for example vectorizing strings
    this._bow = new brain_bow.BagOfWords();
    // _dict is a dictionary build from all the sentences of input data texts
    this._dict = {};
    /* _classes are all the classes from input data text formatted to feed the ANN
     * _classes is like this:
     * 
     *  { apagar_lampara: 0, encender_lampara: 1 }
     */
    this._classes = {};

    /**
     * An array of texts used to train the network
     */
    this._texts = [];

    // The status of the network
    this._status = State.UNTRAINED;

    // Configuration for learning process
    this._configuration = {
        iterations: 3000, // the maximum times to iterate the training data
        errorThresh: 0.0006, // the acceptable error percentage from training data
        log: true, // true to use console.log, when a function is supplied it is used
        logPeriod: 10, // iterations between logging out
        learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
        momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
    };
}

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
BrainText.prototype.setConfiguration = function (config) {
    this._configuration = config;
}

/** 
 * 
*/
BrainText.prototype.getConfiguration = function () {
    return this._configuration
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
 * 
 * returns an object with traindata prepared to be feed in a neural network,
 * the classes as an object wich maps the class name to a number, like this:
 * {apagar_ventilador: 0, encender_ventilador: 1, encender_lampara: 2, apagar_lampara: 3}
 * and the dictionary for BOW.
 */
BrainText.prototype.prepareTrainData = function (traindata) {
    let traindata_for_ann;

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
            input: this._bow.bow(data.text, this._dict),
            output: this._bow.vec_result(
                this._classes[data.label], Object.keys(this._classes).length)
        }
        traindata_for_ann.push(item);
    }

    return traindata_for_ann;
}

BrainText.prototype.setUpdateInfrastructure = function () {
    let { classes, texts, dict }
        = buildClassesAndDict(this._traindata, this._bow);
    this._classes = classes;
    this._texts = texts;
    this._dict = dict;
}

/**
 * Build an array of objects from the input data string
 * each object is like this:
 * {label: "encender_lampara", text:  "enciende la luz"}
 * 
 * The traindata array is cleared when the function start, so 
 * each time this function is called the traindata array is 
 * loaded from scratch.
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
BrainText.prototype.loadTrainDataFromInputDataString = function (inputDataString) {
    // reset traindata vector
    this._traindata = [];
    let inputDataObj = JSON.parse(inputDataString);
    for (const key in inputDataObj) {
        for (const text of inputDataObj[key]) {
            this._traindata.push({ label: key, text: text })
        }
    }
    // now we shuffle traindata
    shuffle(this._traindata);
    this.setUpdateInfrastructure();
}

/**
 * Add new train data. This operation left the network outdate.
 * It must to be trained again to take into account these new data.
 * @param {*} traindata is an array like this:
 * 
 * [{label: 'encender_lampara', text: 'dale a la lamparita'}]
 */
BrainText.prototype.addData = function (traindata) {
    traindata.forEach((data) => {
        this._traindata.forEach((_data) => {
            if (data.text == _data.text) {
                console.log("data repeated!");
                return false;
            }
        })
    });

    this._traindata = this._traindata.concat(traindata);
    this.setUpdateInfrastructure()

    this._status = State.OUTDATED;
    return true;
}

/**
 * Add one new train data. This operation left the network outdate.
 * It must to be trained again to take into account these new data.
 * @param {*} data is an object like this:
 * 
 * {label: 'encender_lampara', text: 'dale a la lamparita'}
 */
BrainText.prototype.addOneData = function (data) {
    this._traindata.forEach((_data) => {
        if (data.text == _data.text) {
            console.log("data repeated!");
            return false;
        }
    })

    this._traindata = this._traindata.concat([data]);
    this.setUpdateInfrastructure();
    this._status = State.OUTDATED;
    return true;
}

/** 
 * Get train data
 */
BrainText.prototype.getTraindata = function () {
    return this._traindata;
}

BrainText.prototype.getState = function () {
    return this._status;
}

BrainText.prototype.getDict = function () {
    return this._dict;
}

/**
 * Train the ANN with input data  JSON string.
 * 
 * Please pay attention!, this function returns a promise.
 * 
 */
BrainText.prototype.train = function () {

    const traindata_for_ann = this.prepareTrainData(this._traindata);

    this._status = State.TRAINING;
    let promise = this._net.trainAsync(traindata_for_ann, this._configuration).then(
        (result) => {
            this._status = State.TRAINED;
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
BrainText.prototype.run = function (entry) {
    if (this._status == State.UNTRAINED) {
        throw "Network UNTRAINED, can't make any prediction!"
    }
    // vectorize as a Bag Of Word
    let term = this._bow.bow(entry, this._dict);
    let predict = this._net.run(term);
    let i = this._bow.maxarg(predict);
    let flippedClasses = {};
    for (let key in this._classes) {
        flippedClasses[this._classes[key]] = key
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
        status: this._status
    }

    return result;
}

/** 
 * Return the model trained as a JSON object
*/
BrainText.prototype.toJSON = function () {
    let model = {
        net: this._net.toJSON(),
        dict: this._dict,
        classes: this._classes,
        texts: this._texts,
        traindata: this._traindata,
    };

    return model;
}

/**
 * 
 * Load a model from a net represented as JSON object (same object obtained
 * with toJSON())
 * 
 * @param {*} json_net is the json object representing the net
 * @param {*} dict is de dictionary build when the net was trained
 * @param {*} classes an object like this {apagar_ventilador: 0, encender_ventilador: 1, encender_lampara: 2, apagar_lampara: 3}
 * @param {*} traindata [{label: 'encender_lampara', text: 'dale a la lamparita'}, {...}]
 */
BrainText.prototype.fromJSON = function (json_model) {
    this._status = State.TRAINED;
    this._dict = json_model.dict;
    this._classes = json_model.classes;
    this._texts = json_model.texts;
    this._traindata = json_model.traindata;
    this._net.fromJSON(json_model.net);
}

module.exports = BrainText;