namespace Core.Models;

/// <summary>
/// Represents the result of a forward pass, including intermediate data.
/// </summary>
public class TrainingResult(double[] inputs, double[] hiddenOutputs, double output)
{
    public double[] Inputs { get; } = inputs;
    public double[] HiddenOutputs { get; } = hiddenOutputs;
    public double Output { get; } = output;
}