namespace Core.Models;

public class ModelData
{
    public List<List<double>> HiddenWeights { get; set; } = [];
    public List<double> HiddenBiases { get; set; } = [];
    public List<double> OutputWeights { get; set; } = [];
    public double OutputBias { get; set; }
}