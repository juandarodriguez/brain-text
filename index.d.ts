export function likely<T>(input: T, net: NeuralNetwork): any;

export interface TrainData {
    label: string,
    text: string
}

export interface Result {
    label: string,
    text: string,
    confidence: number,
    prediction: Object,
    status: string

}

export function loadTrainDataFromInputDataString(inputDataString: string);

export function addData(traindata: TrainData[]);

export function getTrainData(): TrainData[];

export function train();

export function run(entry: string): Result;

