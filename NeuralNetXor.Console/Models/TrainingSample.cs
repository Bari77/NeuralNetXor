namespace NeuralNetXor.Console.Models;

/// <summary>
/// Represents a single training sample with inputs and expected output.
/// </summary>
public class TrainingSample(double[] inputs, double[] expected)
{
    public double[] Inputs { get; set; } = inputs;
    public double[] Expected { get; set; } = expected;
}
