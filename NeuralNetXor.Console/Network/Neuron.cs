namespace NeuralNetXor.Console.Network;

/// <summary>
/// Represents a basic neuron with weighted inputs and a bias.
/// </summary>
public class Neuron
{
    public int InputCount => _inputCount;

    /// <summary>
    /// The last output value (after activation).
    /// </summary>
    public double Output { get; private set; }

    private readonly int _inputCount;
    private readonly double[] _weights;
    private double _bias;
    private static readonly Random _rand = new();

    /// <summary>
    /// Constructor initializing random weights and bias.
    /// </summary>
    public Neuron(int inputCount)
    {
        _inputCount = inputCount;
        _weights = new double[inputCount];
        InitializeWeights();
        _bias = GetRandomWeight();
    }

    private void InitializeWeights()
    {
        for (int i = 0; i < _inputCount; i++)
            _weights[i] = GetRandomWeight();
    }

    private double GetRandomWeight()
    {
        // Random double between -1.0 and 1.0
        return _rand.NextDouble() * 2.0 - 1.0;
    }

    /// <summary>
    /// Performs forward calculation for the neuron.
    /// </summary>
    public double Compute(double[] inputs)
    {
        if (inputs.Length != _inputCount)
            throw new ArgumentException("Input size does not match weight size");

        double sum = 0.0;
        for (int i = 0; i < _inputCount; i++)
            sum += inputs[i] * _weights[i];

        sum += _bias;

        Output = Activation.Sigmoid(sum);
        return Output;
    }

    public double GetWeight(int index)
    {
        return _weights[index];
    }

    public void AdjustWeight(int index, double delta)
    {
        _weights[index] += delta;
    }

    public void AdjustBias(double delta)
    {
        _bias += delta;
    }

    public double GetBias() => _bias;
    public void SetBias(double value) => _bias = value;
    public void SetWeight(int index, double value) => _weights[index] = value;
}