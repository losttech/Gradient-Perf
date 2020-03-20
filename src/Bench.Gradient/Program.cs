using System;
using Gradient;
using TensorFlowNET.Examples;

namespace Bench
{
    class Program
    {
        static void Main(string[] args)
        {
            GradientSetup.EnsureInitialized();
            var sample = new DigitRecognitionCNN();
            sample.Run();
        }
    }
}
