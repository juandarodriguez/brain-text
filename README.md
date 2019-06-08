# BRAIN TEXT

This is a wrapper module over brain.js intended to perform text 
classification.

The module uses *Bag Of Word* strategy and *One Shot Encoding* to map text
and labels (classes) to vectors suitable to be fed in an artificial
neural network of [brain.js](https://github.com/BrainJS/brain.js) library.

# How to use

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
brain_text.setConfiguration(config);
```

The input must be a JSON string with labels and texts. The format must
be like this:

```
const modelJSON = '{"encender_lampara": ["enciende la luz","esto está muy oscuro"],"apagar_lampara": ["apaga la luz","apaga la lámpara"]}';
```

This data can be loaded in network by means of:

```
brain_text.loadTrainDataFromInputDataString(modelJSON);
```

And now the network can be trained:

```
let result = brain_text.train();
```

Since training can be a very intensive CPU operation and may last for a while, the function ``train()`` returns a promise. You only should use the network to perform classification once the trainig process is finished. You can do that by usen ``run()`` function, like this:

```
result.then(() => {
    let r = brain_text.run("encender luz");
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
brain_text.addData([{label: 'encender_lampara', text: 'dale a la lamparita'}]);
```

Now the network is OUTDATED. To take into account these new data, the
network must be trained again:

```
let result = brain_text.train();
```

The file ``test/test.js``shows how to use this module.

