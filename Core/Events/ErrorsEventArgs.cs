namespace Core.Events;

public class ErrorsEventArgs(IReadOnlyList<double> errors) : EventArgs
{
    public IReadOnlyList<double> Errors { get; } = errors;
}