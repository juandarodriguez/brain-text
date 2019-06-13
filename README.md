# BRAIN TEXT

This is a wrapper module over brain.js intended to perform text 
classification.

The module uses *Bag Of Word* strategy and *One Shot Encoding* to map text
and labels (classes) to vectors suitable to be fed in an artificial
neural network of [brain.js](https://github.com/BrainJS/brain.js) library.

# How to use

First create an ``BrainText`` object:

```
let brainText = new BrainText();
```

The network will be trained with this default configuration:

```
{
    iterations: 3000, // the maximum times to iterate the training data
    errorThresh: 0.006, // the acceptable error percentage from training data
    log: true, // true to use console.log, when a function is supplied it is used
    logPeriod: 10, // iterations between logging out
    learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
    momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
};
```

but it can be changed if you want by means of setConfiguration:

```
brainText.setConfiguration(config);
```

The input must be a JSON string with labels and texts. The format must
be like this:

```
const modelJSON = '{"encender_lampara": ["enciende la luz","esto está muy oscuro"],"apagar_lampara": ["apaga la luz","apaga la lámpara"]}';
```

This data can be loaded in network by means of:

```
brainText.loadTrainDataFromInputDataString(modelJSON);
```

And now the network can be trained:

```
let result = brainText.train();
```

Since training can be a very intensive CPU operation and may last for a while, the function ``train()`` returns a promise. You only should use the network to perform classification once the trainig process is finished. You can do that by using ``run()`` function, like this:

```
result.then(() => {
    let r = brainText.run("encender luz");
    console.log(r);
});
```

The result has this aspect:

```
{ text: 'encender luz',
  label: 'encender_lampara',
  confidence: 0.7557198405265808,
  prediction: 
   { encender_lampara: 0.7557198405265808,
     apagar_lampara: 0.24401657283306122 },
     status: 'TRAINED' 
   }
```

After traininig, new data can be added to train data:

```
brainText.addData([
    {label: 'encender_lampara', text: 'dale a la lamparita'},
    {label: 'apagar_lampara', text: 'quita la lamparita'}
    ]);
```

If you want to add just a train data:

```
brainText.addOneData({label: 'encender_lampara', text: 'dale a la lamparita'});
```

Now the network is OUTDATED. To take into account these new data, the
network must be trained again:

```
let result = brainText.train();
```

We can obtain a JSON model 

```
let jsonModel = brainText.toJSON();
```

This json model can be  used to be loaded as model in a ``BrainText`` object:

```
let jsonModel = brainText.toJSON();

let brainText2 = new BrainText();

brainText2.fromJSON(jsonModel);
```

Once a new BrainText object has been created by means of ``fromJSON``, you can add new data 
and train again to obtain a better model.

This JSON model can be also used in order to create a neural network with ``brain.js`` library:

```
let jsonModel = brainText.train();

const brain = require('brain.js');

let net = new new brain.NeuralNetwork();

net.fromJSON(jsonModel.net);
```


The file ``test/test.js``shows how to use this module.

