using System;
using Gradient;
using TensorFlowNET.Examples;

namespace Bench
{
    static class GradientBenchProgram
    {
        static void Main()
        {
            GradientEngine.UseEnvironmentFromVariable();
            GradientSetup.EnsureInitialized();
            var sample = new DigitRecognitionCNN();
            sample.Run();
        }
    }
}
