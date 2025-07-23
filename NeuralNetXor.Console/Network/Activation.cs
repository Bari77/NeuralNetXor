namespace NeuralNetXor.Console.Network;

/// <summary>
/// Static class for activation functions.
/// </summary>
public static class Activation
{
    /// <summary>
    /// Sigmoid function: squashes a value between 0 and 1.
    /// </summary>
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    /// <summary>
    /// Derivative of the sigmoid function (for learning).
    /// </summary>
    public static double SigmoidDerivative(double x)
    {
        double sigmoid = Sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }
}
