const brain_text = require('../index');
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
    beforeEach(function () {
        brain_text.setConfiguration(config)
        brain_text.loadTrainDataFromInputDataString(modelJSON);
        return brain_text.train();
    });

    it("Configuration correct", function () {
        let c = brain_text.getConfiguration();
        expect(c).toEqual(jasmine.objectContaining(config));
    });

    it("Get train data", function () {
        let td = brain_text.getTrainData();
        expect(td).toEqual(jasmine.arrayContaining(
            [{ label: "encender_lampara", text: "enciende la luz" }]
        ))
    });

    it("Run model", function () {
        let r = brain_text.run("enciende la luz");
        expect(r).toEqual(jasmine.objectContaining({
            status: "TRAINED",
            label: "encender_lampara"
        }));
    });

    it("Add data", function () {
        brain_text.addData([
            { label: "encender_lampara", text: "dale a la lamparita" },
            { label: "apagar_lampara", text: "quita la lamparita" }
        ]);
        expect(brain_text.getTrainData()).toEqual(jasmine.arrayContaining(
            [{ label: "encender_lampara", text: "dale a la lamparita" }]
        ));

        expect(brain_text.getTrainData()).toEqual(jasmine.arrayContaining(
            [{ label: "apagar_lampara", text: "quita la lamparita" }]
        ));
    });

    it("Add just one data", function(){
        brain_text.addOneData({ label: "encender_lampara", text: "dale a la lucecita" });
        expect(brain_text.getTrainData()).toEqual(jasmine.arrayContaining(
            [{ label: "encender_lampara", text: "dale a la lucecita" }]
        ));
    })
});

describe("Running after retrain", function () {
    beforeEach(function () {
        brain_text.setConfiguration(config)
        brain_text.loadTrainDataFromInputDataString(modelJSON);
        return brain_text.train().then((r) => {
            brain_text.addData([{ label: "encender_lampara", text: "dale a la lamparita" }]);
            return brain_text.train();
        });
    });

    it("Run model after retrained", function () {
        let r = brain_text.run("dale a la lamparita");
        expect(r).toEqual(jasmine.objectContaining(
            { label: "encender_lampara", status: "TRAINED" }
        ));
        //console.log(r);
        expect(r.confidence > 0.7).toBe(true);
    });

});

// describe("Export and import JSON", function () {
//     let json_net = null;
//     beforeEach(function () {
//         brain_text.setConfiguration(config)
//         brain_text.loadTrainDataFromInputDataString(modelJSON);
//         json_net = brain_text.toJSON();
//         return brain_text.train();
//     });

//     it("Load network from JSON and run model", function () {
//         brain_text.fromJSON(json_net);
//         let r = brain_text.run("encender luz");
//         expect(r).toEqual(jasmine.objectContaining({
//             status: "TRAINED",
//             label: "encender_lampara"
//         }));
//     });

// });