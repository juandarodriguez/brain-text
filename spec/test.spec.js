const BrainText  = require('../index');

console.log(BrainText);

const modelJSON = '{"encender_lampara": ["enciende la luz","esto está muy oscuro"],"apagar_lampara": ["apaga la luz","apaga la lámpara"]}';

const config = {
    iterations: 3000, // the maximum times to iterate the training data
    errorThresh: 0.006, // the acceptable error percentage from training data
    log: true, // true to use console.log, when a function is supplied it is used
    logPeriod: 10, // iterations between logging out
    learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
    momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
};

describe("Basic operations", function () {
    let brainText;
    beforeEach(function () {
        brainText = new BrainText();
        brainText.setConfiguration(config)
        brainText.loadTrainDataFromInputDataString(modelJSON);
        return brainText.train();
    });

    it("Configuration correct", function () {
        let c = brainText.getConfiguration();
        expect(c).toEqual(jasmine.objectContaining(config));
    });

    it("Get train data", function () {
        let td = brainText.getTraindata();
        expect(td).toEqual(jasmine.arrayContaining(
            [{ label: "encender_lampara", text: "enciende la luz" }]
        ))
    });

    it("Run model", function () {
        let r = brainText.run("enciende la luz");
        expect(r).toEqual(jasmine.objectContaining({
            status: "TRAINED",
            label: "encender_lampara"
        }));
    });

    it("Add data", function () {
        brainText.addData([
            { label: "encender_lampara", text: "dale a la lamparita" },
            { label: "apagar_lampara", text: "quita la lamparita" }
        ]);
        expect(brainText.getTraindata()).toEqual(jasmine.arrayContaining(
            [{ label: "encender_lampara", text: "dale a la lamparita" }]
        ));

        expect(brainText.getTraindata()).toEqual(jasmine.arrayContaining(
            [{ label: "apagar_lampara", text: "quita la lamparita" }]
        ));
    });

    it("Add just one data", function(){
        brainText.addOneData({ label: "encender_lampara", text: "dale a la lucecita" });
        expect(brainText.getTraindata()).toEqual(jasmine.arrayContaining(
            [{ label: "encender_lampara", text: "dale a la lucecita" }]
        ));
    })
});

describe("Running after retrain", function () {
    let brainText;
    beforeEach(function () {
        brainText = new BrainText();
        brainText.setConfiguration(config)
        brainText.loadTrainDataFromInputDataString(modelJSON);
        return brainText.train().then((r) => {
            brainText.addData([{ label: "encender_lampara", text: "dale a la lamparita" }]);
            return brainText.train();
        });
    });

    it("Run model after retrained", function () {
        let r = brainText.run("dale a la lamparita");
        expect(r).toEqual(jasmine.objectContaining(
            { label: "encender_lampara", status: "TRAINED" }
        ));
        //console.log(r);
        expect(r.confidence > 0.7).toBe(true);
    });

});

describe("Export and import JSON", function () {
    let json_net;
    let brainText;
    beforeEach(function () {
        brainText = new BrainText();
        brainText.setConfiguration(config)
        brainText.loadTrainDataFromInputDataString(modelJSON);
        return brainText.train().then(() => {
            json_net = brainText.toJSON();
            return true;
        });
    });

    it("Load network from JSON and run model", function () {
        let brainText2 = new BrainText();
        brainText2.fromJSON(json_net);
        let r = brainText2.run("encender luz");
        expect(r).toEqual(jasmine.objectContaining({
            status: "TRAINED",
            label: "encender_lampara"
        }));
    });

});

describe("Create network just adding data", function(){
    let brainText;
    beforeEach(function () {
        brainText = new BrainText();
        brainText.setConfiguration(config)
        brainText.addData([
            {label: 'encender_lampara', text: "enciende la luz"},
            {label: 'apagar_lampara', text: "apaga la luz"}
        ]);
        return brainText.train();
    });

    it("Run model trained", function () {
        let r = brainText.run("enciende la luz");
        expect(r).toEqual(jasmine.objectContaining(
            { label: "encender_lampara", status: "TRAINED" }
        ));
        //console.log(r);
        expect(r.confidence > 0.7).toBe(true);
    });
});