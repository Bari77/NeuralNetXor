namespace Core.Events;

public class TrainingProgressEventArgs(int epoch, double error) : EventArgs
{
    public int Epoch { get; } = epoch;
    public double Error { get; } = error;
}
