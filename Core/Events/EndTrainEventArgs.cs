namespace Core.Events;

public class EndTrainEventArgs(double confidenceThreshold, int epoch) : EventArgs
{
    public double ConfidenceThreshold { get; } = confidenceThreshold;
    public int Epoch { get; } = epoch;
}