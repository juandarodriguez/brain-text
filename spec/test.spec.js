const brain_text = require('../index');
const modelJSON = '{"encender_lampara": ["enciende la luz","esto está muy oscuro"],"apagar_lampara": ["apaga la luz","apaga la lámpara"]}';

const config = {
    iterations: 3000, // the maximum times to iterate the training data
    errorThresh: 0.0006, // the acceptable error percentage from training data
    log: true, // true to use console.log, when a function is supplied it is used
    logPeriod: 10, // iterations between logging out
    learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
    momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
};

describe("Change configuration", function () {
    it("Configuration correct", function () {
        expect(brain_text.setConfiguration(config)).toBe(true);
    });
});

