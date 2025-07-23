using Core.Models;
using Core.Network;
using System.Globalization;
using System.Text.Json;

namespace Xor.ConsoleApp;

internal class Program
{
    private static readonly string MODEL_PATH = "model.json";
    private static NeuralNetwork? _network;

    /// <summary>
    /// Point d’entrée principal du programme.
    /// Gère l’intro, la sélection du modèle, l'entraînement éventuel, puis démarre l’interaction utilisateur.
    /// </summary>
    private static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.Title = "[NeuralNet] XOR ConsoleApp";

        ShowIntro();

        AskModelPreference(out bool resetRequested);
        SetupNetwork();

        if (!resetRequested && File.Exists(MODEL_PATH))
        {
            LoadModel();
        }
        else
        {
            TrainAndSaveModel();
        }

        RunInteractionLoop();
    }

    /// <summary>
    /// Affiche l’introduction expliquant la porte logique XOR.
    /// </summary>
    private static void ShowIntro()
    {
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
    ");
    }

    /// <summary>
    /// Demande à l’utilisateur s’il souhaite utiliser le modèle existant ou le réinitialiser.
    /// </summary>
    /// <param name="resetRequested">Sortie booléenne indiquant si un réentraînement est requis.</param>
    private static void AskModelPreference(out bool resetRequested)
    {
        Console.Write("Souhaites-tu utiliser le modèle existant si présent ? (O/n) : ");
        string? answer = Console.ReadLine();
        resetRequested = answer?.Trim().ToLower() == "n";
    }

    /// <summary>
    /// Initialise le réseau de neurones et configure les callbacks d'affichage.
    /// </summary>
    private static void SetupNetwork()
    {
        _network = new NeuralNetwork(inputCount: 2, hiddenCount: 2);

        _network.TrainingProgress += (sender, args) =>
        {
            Console.WriteLine($"Epoch {args.Epoch} - Total Error: {args.Error:F6}");
        };

        _network.EndTrain += (sender, args) =>
        {
            Console.WriteLine($"\n✅ Le réseau a atteint un niveau de confiance ≥ {args.ConfidenceThreshold:P0} à l'epoch {args.Epoch} !");
        };

        _network.ErrorsGenerated += (sender, args) =>
        {
            const int graphHeight = 10;
            const int graphWidth = 50;

            var errors = args.Errors;
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
        };
    }

    /// <summary>
    /// Entraîne le réseau sur le jeu XOR, puis sauvegarde le modèle dans un fichier JSON.
    /// </summary>
    private static void TrainAndSaveModel()
    {
        Console.WriteLine("🔧 Entraînement du réseau...\n");

        var trainingSet = new List<TrainingSample>
        {
            new([0, 0], [0]),
            new([0, 1], [1]),
            new([1, 0], [1]),
            new([1, 1], [0])
        };

        _network!.Train(trainingSet, maxEpochs: 200000, learningRate: 0.1, 0.95);

        Console.WriteLine("\n✅ Entraînement terminé !");
        var model = _network.ExportModel();
        var json = JsonSerializer.Serialize(model, options: new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(MODEL_PATH, json);
    }

    /// <summary>
    /// Charge un modèle de réseau préalablement sauvegardé à partir du fichier JSON.
    /// </summary>
    private static void LoadModel()
    {
        Console.WriteLine("📦 Chargement du modèle existant...");
        var json = File.ReadAllText(MODEL_PATH);
        var model = JsonSerializer.Deserialize<ModelData>(json);
        _network!.ImportModel(model!);
        Console.WriteLine("✅ Modèle chargé !");
    }

    /// <summary>
    /// Démarre une boucle interactive dans laquelle l'utilisateur peut tester le réseau.
    /// </summary>
    private static void RunInteractionLoop()
    {
        Console.WriteLine("\nTu peux maintenant tester ton réseau avec des entrées (0 ou 1).\n");

        while (true)
        {
            Console.Write("👉 Entrée A (0 ou 1) : ");
            double a = ReadInput();

            Console.Write("👉 Entrée B (0 ou 1) : ");
            double b = ReadInput();

            double[] inputs = [a, b];
            double output = _network!.Compute(inputs);

            Console.WriteLine($"→ Sortie du réseau : {output:F0} ({output:F4})");
            Console.WriteLine("---\n");
        }
    }

    /// <summary>
    /// Lit et valide une entrée utilisateur binaire (0 ou 1).
    /// </summary>
    /// <returns>La valeur entière saisie par l'utilisateur (0 ou 1).</returns>
    private static double ReadInput()
    {
        while (true)
        {
            var input = Console.ReadLine();
            if (input == "0" || input == "1")
                return double.Parse(input!, CultureInfo.InvariantCulture);

            Console.Write("❌ Veuillez entrer 0 ou 1 : ");
        }
    }
}