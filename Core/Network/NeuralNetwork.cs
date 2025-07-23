using Core.Events;
using Core.Models;

namespace Core.Network;

/// <summary>
/// Represents a basic feedforward neural network with 1 hidden layer.
/// </summary>
public class NeuralNetwork
{
    public event EventHandler<TrainingProgressEventArgs>? TrainingProgress;
    public event EventHandler<EndTrainEventArgs>? EndTrain;
    public event EventHandler<ErrorsEventArgs>? ErrorsGenerated;

    private readonly Neuron[] _hiddenLayer;
    private readonly Neuron _outputNeuron;

    /// <summary>
    /// Creates a neural network with fixed sizes.
    /// </summary>
    public NeuralNetwork(int inputCount, int hiddenCount)
    {
        _hiddenLayer = new Neuron[hiddenCount];
        for (int i = 0; i < hiddenCount; i++)
            _hiddenLayer[i] = new Neuron(inputCount);

        _outputNeuron = new Neuron(hiddenCount);
    }

    /// <summary>
    /// Computes the network output for the given inputs.
    /// </summary>
    public double Compute(double[] inputs)
    {
        double[] hiddenOutputs = new double[_hiddenLayer.Length];

        for (int i = 0; i < _hiddenLayer.Length; i++)
            hiddenOutputs[i] = _hiddenLayer[i].Compute(inputs);

        return _outputNeuron.Compute(hiddenOutputs);
    }

    /// <summary>
    /// Computes the output and returns a TrainingResult with internals.
    /// (For backpropagation use)
    /// </summary>
    public TrainingResult ForwardWithInternals(double[] inputs)
    {
        double[] hiddenOutputs = new double[_hiddenLayer.Length];

        for (int i = 0; i < _hiddenLayer.Length; i++)
            hiddenOutputs[i] = _hiddenLayer[i].Compute(inputs);

        double finalOutput = _outputNeuron.Compute(hiddenOutputs);

        return new TrainingResult(inputs, hiddenOutputs, finalOutput);
    }

    public void Train(List<TrainingSample> trainingData, int maxEpochs, double learningRate, double confidenceThreshold)
    {
        var errors = new List<double>();

        for (int epoch = 1; epoch <= maxEpochs; epoch++)
        {
            double totalError = 0.0;

            foreach (var sample in trainingData)
            {
                var result = ForwardWithInternals(sample.Inputs);
                double outputError = sample.Expected[0] - result.Output;
                double outputDelta = outputError * result.Output * (1 - result.Output);
                totalError += Math.Pow(outputError, 2);

                // Ajustement du neurone de sortie
                for (int i = 0; i < _outputNeuron.InputCount; i++)
                {
                    double delta = learningRate * outputDelta * result.HiddenOutputs[i];
                    _outputNeuron.AdjustWeight(i, delta);
                }
                _outputNeuron.AdjustBias(learningRate * outputDelta);

                // Ajustement des neurones cachés
                for (int i = 0; i < _hiddenLayer.Length; i++)
                {
                    double hiddenOutput = result.HiddenOutputs[i];
                    double hiddenDelta = outputDelta * _outputNeuron.GetWeight(i) * hiddenOutput * (1 - hiddenOutput);

                    for (int j = 0; j < _hiddenLayer[i].InputCount; j++)
                    {
                        double delta = learningRate * hiddenDelta * result.Inputs[j];
                        _hiddenLayer[i].AdjustWeight(j, delta);
                    }
                    _hiddenLayer[i].AdjustBias(learningRate * hiddenDelta);
                }
            }

            errors.Add(totalError);
            TrainingProgress?.Invoke(this, new TrainingProgressEventArgs(epoch, totalError));

            if (HasLearnedAll(trainingData, confidenceThreshold))
            {
                EndTrain?.Invoke(this, new EndTrainEventArgs(confidenceThreshold, epoch));
                break;
            }
        }

        ErrorsGenerated?.Invoke(this, new ErrorsEventArgs(errors));
    }

    private bool HasLearnedAll(List<TrainingSample> trainingSet, double confidenceThreshold)
    {
        foreach (var sample in trainingSet)
        {
            double predicted = Compute(sample.Inputs);
            double expected = sample.Expected[0];

            if (expected == 1.0 && predicted < confidenceThreshold)
                return false;
            if (expected == 0.0 && predicted > (1.0 - confidenceThreshold))
                return false;
        }
        return true;
    }

    public ModelData ExportModel()
    {
        return new ModelData
        {
            HiddenWeights = _hiddenLayer
                .Select(n => Enumerable.Range(0, n.InputCount).Select(n.GetWeight).ToList())
                .ToList(),
            HiddenBiases = _hiddenLayer
                .Select(n => n.GetBias())
                .ToList(),
            OutputWeights = Enumerable.Range(0, _outputNeuron.InputCount)
                .Select(_outputNeuron.GetWeight)
                .ToList(),
            OutputBias = _outputNeuron.GetBias()
        };
    }

    public void ImportModel(ModelData model)
    {
        for (int i = 0; i < _hiddenLayer.Length; i++)
        {
            for (int j = 0; j < _hiddenLayer[i].InputCount; j++)
                _hiddenLayer[i].SetWeight(j, model.HiddenWeights[i][j]);

            _hiddenLayer[i].SetBias(model.HiddenBiases[i]);
        }

        for (int i = 0; i < _outputNeuron.InputCount; i++)
            _outputNeuron.SetWeight(i, model.OutputWeights[i]);

        _outputNeuron.SetBias(model.OutputBias);
    }

}
