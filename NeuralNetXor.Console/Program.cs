using NeuralNetXor.Console.Models;
using NeuralNetXor.Console.Network;
using System.Globalization;

Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.WriteLine(@"
╔════════════════════════════════════════════════════╗
║      🧠 Réseau de Neurones : Porte logique XOR     ║
╚════════════════════════════════════════════════════╝

Une porte logique XOR (exclusive OR) renvoie :
 → 1 si les deux entrées sont différentes
 → 0 si elles sont identiques

  Entrée A   Entrée B   Résultat attendu
  ─────────  ─────────  ─────────────────
     0          0              0
     0          1              1
     1          0              1
     1          1              0

L’objectif de ce programme est de faire apprendre cette logique
à un réseau de neurones sans lui dire les règles à l’avance.

Il va ""deviner"" les bons résultats par ajustement automatique.

[APPUYER SUR UNE TOUCHE POUR COMMENCER L'ENTRAINEMENT]
");
Console.ReadKey();

var network = new NeuralNetwork(inputCount: 2, hiddenCount: 2)
{
    TrainingProgress = (epoch, error) =>
    {
        Console.WriteLine($"Epoch {epoch} - Total Error: {error:F6}");
    },
    EndTrain = (confidenceThreshold, epoch) =>
    {
        Console.WriteLine($"\n✅ Le réseau a atteint un niveau de confiance ≥ {confidenceThreshold:P0} à l'epoch {epoch} !");
    },
    GetErrors = (errors) =>
    {
        const int graphHeight = 10;
        const int graphWidth = 50;

        double maxError = errors.Max();
        double minError = errors.Min();
        double range = maxError - minError;

        Console.WriteLine("\n📈 Courbe d'erreur :");
        for (int y = graphHeight - 1; y >= 0; y--)
        {
            double threshold = minError + (range * y / (graphHeight - 1));
            string line = "";

            for (int x = 0; x < graphWidth; x++)
            {
                int index = x * errors.Count / graphWidth;
                if (index < errors.Count && errors[index] >= threshold)
                    line += "█";
                else
                    line += " ";
            }

            Console.WriteLine(line);
        }

        Console.WriteLine(new string('─', graphWidth));
    }
};

// Entraîne le réseau
Console.WriteLine("🔧 Entraînement du réseau...\n");

// Crée des données d'entraînement pour le XOR
var trainingSet = new List<TrainingSample>
{
    new([0, 0], [0]),
    new([0, 1], [1]),
    new([1, 0], [1]),
    new([1, 1], [0])
};

network.Train(trainingSet, maxEpochs: 200000, learningRate: 0.1, 0.95);

Console.WriteLine("\n✅ Entraînement terminé !");
Console.WriteLine("Tu peux maintenant tester ton réseau avec des entrées (0 ou 1).\n");

// Boucle principale
while (true)
{
    Console.Write("👉 Entrée A (0 ou 1) : ");
    double a = ReadInput();

    Console.Write("👉 Entrée B (0 ou 1) : ");
    double b = ReadInput();

    double[] inputs = [a, b];

    double output = network.Compute(inputs);

    Console.WriteLine($"-> Sortie du réseau : {output:F0} ({output:F4})");
    Console.WriteLine("---\n");
}

static double ReadInput()
{
    while (true)
    {
        var input = Console.ReadLine();
        if (input == "0" || input == "1")
            return double.Parse(input!, CultureInfo.InvariantCulture);

        Console.Write("❌ Veuillez entrer 0 ou 1 : ");
    }
}
